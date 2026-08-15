"""
US Historical Economic Data Collection (1960-2025)
For State Response Function Calibration

Data sources:
- FRED (Federal Reserve Economic Data) - macroeconomic series
- Census Bureau - wage percentiles, Gini
- Tax Foundation - tax rates
- CNTSDA / BLS - strike/protest data
- SSA, USDA, HHS - redistribution program spending

Output: DataFrame indexed by quarter, ready for state response function fitting
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
from data.format.secret import FRED_API_KEY
import warnings
warnings.filterwarnings('ignore')

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    print("⚠ fredapi not installed. Install with: pip install fredapi")
    FRED_AVAILABLE = False

class ManualDataTranslator:
    """Ingests manually collected data files with robust error handling."""
    
    def __init__(self, data_dir="data/raw/", quarterly_index=None):
        self.data_dir = Path(data_dir)
        self.quarterly_index = quarterly_index
        self.translated_series = {}
        self.diagnostics = {}
    
    def _align_to_quarterly(self, series, series_name, fill_method='ffill', 
                           fill_gaps_with_zero=False):
        """Align series to quarterly frequency."""
        if len(series) == 0:
            print(f"  ⚠ {series_name}: Empty series")
            return pd.Series(np.nan, index=self.quarterly_index)
        
        # Resample to quarterly
        if series.index.freq is None or 'Q' not in str(series.index.freq):
            quarterly = series.resample('QE').mean()
        else:
            quarterly = series
        
        # Align to target index using reindex (handles dates outside range)
        aligned = pd.Series(np.nan, index=self.quarterly_index)
        
        # Find overlapping dates and align
        common_dates = quarterly.index.intersection(self.quarterly_index)
        if len(common_dates) > 0:
            aligned.loc[common_dates] = quarterly.loc[common_dates]
        
        # Forward-fill or interpolate
        if fill_method == 'ffill':
            aligned = aligned.ffill()
        elif fill_method == 'interpolate':
            aligned = aligned.interpolate(method='linear')
        
        # Fill pre-policy period with 0
        if fill_gaps_with_zero:
            first_value_idx = aligned.first_valid_index()
            if first_value_idx is not None:
                pre_policy_mask = self.quarterly_index < first_value_idx
                aligned[pre_policy_mask] = 0.0
        
        self.diagnostics[series_name] = {
            'n_original': len(series),
            'n_aligned': aligned.notna().sum(),
            'date_range': f"{self.quarterly_index.min().date()} to {self.quarterly_index.max().date()}",
            'coverage_pct': (aligned.notna().sum() / len(aligned)) * 100
        }
        
        return aligned
    
    def ingest_and_merge_strike_data(self, annual_file, detailed_file, 
                                     annual_date_col, annual_workers_col, annual_days_col,
                                     detailed_date_col, detailed_workers_col, detailed_days_col,
                                     annual_sheet="Annual listing", detailed_sheet="Monthly listing",
                                     cutover_year=1988):
        """Ingest and merge two strike datasets."""
        filepath_annual = self.data_dir / annual_file
        filepath_detailed = self.data_dir / detailed_file
        
        try:
            # ===== INGEST ANNUAL DATA =====
            df_annual = pd.read_excel(filepath_annual, sheet_name=annual_sheet)
            
            # Parse year column
            df_annual['year'] = pd.to_numeric(df_annual[annual_date_col], errors='coerce')
            
            # Clean numeric columns
            df_annual[annual_workers_col] = pd.to_numeric(
                df_annual[annual_workers_col].astype(str).str.replace(',', ''), 
                errors='coerce'
            )
            df_annual[annual_days_col] = pd.to_numeric(
                df_annual[annual_days_col].astype(str).str.replace(',', ''), 
                errors='coerce'
            )
            
            # Filter to pre-cutover
            df_annual = df_annual[df_annual['year'] < cutover_year].dropna(subset=['year'])
            
            annual_agg = df_annual.groupby('year').agg({
                annual_workers_col: 'sum',
                annual_days_col: 'sum'
            }).reset_index()
            
            annual_agg.rename(columns={
                annual_workers_col: 'workers_affected',
                annual_days_col: 'days_idle'
            }, inplace=True)
            
            print(f"  ✓ {annual_file} (pre-{cutover_year}) → {len(annual_agg)} annual records")
            
            # ===== INGEST DETAILED DATA =====
            df_detailed = pd.read_excel(filepath_detailed, sheet_name=detailed_sheet)
            
            # Parse dates - try multiple formats
            df_detailed['date'] = pd.to_datetime(df_detailed[detailed_date_col], errors='coerce')
            df_detailed['year'] = df_detailed['date'].dt.year
            
            # Remove rows without valid dates
            df_detailed = df_detailed[df_detailed['year'].notna()]
            
            # Clean numeric columns
            df_detailed[detailed_workers_col] = pd.to_numeric(
                df_detailed[detailed_workers_col].astype(str).str.replace(',', ''), 
                errors='coerce'
            )
            df_detailed[detailed_days_col] = pd.to_numeric(
                df_detailed[detailed_days_col].astype(str).str.replace(',', ''), 
                errors='coerce'
            )
            
            # Aggregate to annual, filter to >= cutover_year
            detailed_agg = df_detailed[df_detailed['year'] >= cutover_year].groupby('year').agg({
                detailed_workers_col: 'sum',
                detailed_days_col: 'sum'
            }).reset_index()
            
            detailed_agg.rename(columns={
                detailed_workers_col: 'workers_affected',
                detailed_days_col: 'days_idle'
            }, inplace=True)
            
            print(f"  ✓ {detailed_file} (>={cutover_year}) → {len(detailed_agg)} annual records")
            
            # ===== MERGE =====
            merged_agg = pd.concat([annual_agg, detailed_agg], ignore_index=True)
            merged_agg = merged_agg.sort_values('year').drop_duplicates(subset='year')
            
            print(f"  ✓ Merged: {len(merged_agg)} total annual records (cutover at {cutover_year})")
            
            # Convert to quarterly series (repeat annual value for all 4 quarters)
            workers_series = pd.Series(
                merged_agg['workers_affected'].values,
                index=pd.to_datetime(merged_agg['year'], format='%Y')
            )
            days_series = pd.Series(
                merged_agg['days_idle'].values,
                index=pd.to_datetime(merged_agg['year'], format='%Y')
            )
            
            # Align to quarterly
            workers_aligned = self._align_to_quarterly(
                workers_series, 'workers_affected',
                fill_method='ffill', fill_gaps_with_zero=False
            )
            days_aligned = self._align_to_quarterly(
                days_series, 'days_idle',
                fill_method='ffill', fill_gaps_with_zero=False
            )
            
            self.translated_series.setdefault('United States', {})['workers_affected'] = workers_aligned
            self.translated_series.setdefault('United States', {})['days_idle'] = days_aligned
            
            return (workers_aligned, days_aligned)
        
        except Exception as e:
            print(f"  ✗ Strike merge failed: {str(e)[:80]}")
            import traceback
            traceback.print_exc()
            return None
    
    def ingest_cntsdata_protests(self, filename, sheet_name, year_col, 
                                 value_cols, series_name, country_filter='United States'):
        """
        Ingest CNTSDATA protest events (multi-country dataset).
        Filters to US, sums specified event types.
        """
        filepath = self.data_dir / filename
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            # Filter to US
            if 'Country' in df.columns:
                df = df[df['Country'] == country_filter].copy()
            elif 'country_name' in df.columns:
                df = df[df['country_name'] == country_filter].copy()
            
            if len(df) == 0:
                print(f"  ⚠ {filename}: No US records found")
                return None
            
            # Parse year
            df['year'] = pd.to_numeric(df[year_col], errors='coerce')
            
            # Sum event types
            for col in value_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Aggregate by year
            agg_dict = {col: 'sum' for col in value_cols if col in df.columns}
            annual_agg = df.groupby('year')[list(agg_dict.keys())].sum().reset_index()
            
            # Sum all events into single series
            annual_agg['total_events'] = annual_agg[[col for col in value_cols if col in annual_agg.columns]].sum(axis=1)
            
            # Convert to series
            series = pd.Series(
                annual_agg['total_events'].values,
                index=pd.to_datetime(annual_agg['year'], format='%Y')
            )
            
            aligned = self._align_to_quarterly(series, series_name,
                                               fill_method='ffill', fill_gaps_with_zero=False)
            self.translated_series.setdefault('United States', {})[series_name] = aligned
            
            print(f"  ✓ {filename} (US only) → {series_name} ({len(series)} years)")
            return aligned
        
        except Exception as e:
            print(f"  ✗ {filename}: {str(e)[:50]}")
            return None

    def ingest_and_adjust_fiscal_year(self, filename, date_col=None, value_col=None, 
                                    series_name=None, file_type='xlsx', 
                                    sheet_name=None, encoding='utf-8', skip_rows=0,
                                    is_fiscal_year=True, country="United States",
                                    wide_year_columns=False, country_col="Country Name",
                                    country_filter=None):
        """
        Ingest annual data (fiscal or calendar year), adjust if FY, then align to quarterly.
        Single flow with consistent diagnostics from _align_to_quarterly().
        
        Args:
            filename: str
            date_col: str, column name with dates. Not used when
                wide_year_columns=True (see below).
            value_col: str or list, column(s) with values. Not used when
                wide_year_columns=True.
            series_name: str
            file_type: 'xlsx' or 'csv'
            sheet_name: str (only if xlsx)
            encoding: str (only if csv)
            skip_rows: int (only if csv)
            country: str, defaults to "United States" so existing calls are
                unaffected. Stored in self.diagnostics[series_name] so the
                final long-format assembly step knows which country each
                series belongs to -- doesn't change any ingestion behavior.
            wide_year_columns: bool, default False. Set True for World
                Bank-style files where there's no date_col at all -- each
                YEAR is its own column (e.g. "1990", "1991", ... "2023"),
                and rows are countries. When True, date_col/value_col are
                ignored; instead the row matching country_filter in
                country_col is located, every column that parses as a
                bare 4-digit year is treated as one (year, value) point,
                and everything downstream (FY adjustment, quarterly
                alignment) runs exactly as it does for the normal
                long/date_col format -- this only changes how the raw
                (year, value) pairs get extracted.
            country_col: str, column holding country names. Only used
                when wide_year_columns=True. World Bank exports typically
                use "Country Name".
            country_filter: str, matched as a case-insensitive SUBSTRING
                against country_col (e.g. "Egypt" will match World Bank's
                "Egypt, Arab Rep."). Takes the first match, so make sure
                your file is trimmed enough that the substring can't hit
                more than one row. Required when wide_year_columns=True.
        
        Returns:
            pd.Series aligned to quarterly (or None if failed)
        """
        filepath = self.data_dir / filename
        
        try:
            # ===== INGEST =====
            if file_type == 'xlsx':
                df = pd.read_excel(filepath, sheet_name=sheet_name)
            else:  # csv
                actual_encoding = encoding
                try:
                    df = pd.read_csv(filepath, encoding=encoding, skiprows=skip_rows)
                except UnicodeDecodeError:
                    for enc in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(filepath, encoding=enc, skiprows=skip_rows)
                            actual_encoding = enc
                            break
                        except:
                            continue
                    else:
                        print(f"  ✗ {filename}: Could not decode")
                        return None
            
            # Clean columns
            df.columns = [str(c).strip() for c in df.columns]

            if wide_year_columns:
                # ===== WORLD BANK-STYLE WIDE FORMAT =====
                # No date_col -- extract (year, value) pairs directly from
                # the row matching country_filter, then feed the same
                # series_annual variable into the unchanged FY/quarterly
                # logic below.
                if country_col not in df.columns:
                    print(f"  ✗ {filename}: country_col '{country_col}' not found "
                          f"(have: {list(df.columns)[:8]}...)")
                    return None
                if country_filter is None:
                    print(f"  ✗ {filename}: wide_year_columns=True requires country_filter")
                    return None

                row = df[df[country_col].astype(str).str.contains(
                    re.escape(country_filter), case=False, na=False
                )]
                if len(row) == 0:
                    print(f"  ✗ {filename}: no row where {country_col} contains '{country_filter}'")
                    return None
                row = row.iloc[0]

                year_cols = [c for c in df.columns if re.match(r'^\d{4}$', str(c).strip())]
                dates, data_values = [], []
                for yc in year_cols:
                    v = pd.to_numeric(str(row[yc]).replace(',', '').replace('%', ''),
                                       errors='coerce')
                    if pd.notna(v):
                        dates.append(pd.Timestamp(f"{yc}-01-01"))
                        data_values.append(v)

                series_annual = pd.Series(data_values, index=pd.DatetimeIndex(dates))
                if len(series_annual) == 0:
                    print(f"  ✗ {filename}: no valid year columns for '{country_filter}'")
                    return None
                print(f"  ✓ {filename} (wide format, {country_filter}) → "
                      f"{len(series_annual)} year columns found")

            else:
                # ===== ORIGINAL LONG FORMAT (date_col + value_col) =====
                # Parse values (handle % and commas)
                if isinstance(value_col, list):
                    data_values = df[value_col].sum(axis=1, skipna=True).values
                else:
                    values_raw = df[value_col].astype(str).str.replace('%', '').str.replace(',', '')
                    data_values = pd.to_numeric(values_raw, errors='coerce').values
                
                # Parse dates
                raw_col = df[date_col]
                if pd.api.types.is_numeric_dtype(raw_col) or raw_col.astype(str).str.match(r'^\d{4}$').all():
                    # bare year integers (e.g. 1970, 1971...) -> treat as Jan 1 of that year
                    dates = pd.to_datetime(raw_col.astype(int).astype(str) + '-01-01', errors='coerce')
                else:
                    dates = pd.to_datetime(raw_col, errors='coerce')
                
                # Remove NaN rows
                mask = dates.notna() & pd.Series(data_values).notna()
                dates = dates[mask]
                data_values = pd.Series(data_values)[mask].values
                
                series_annual = pd.Series(data_values, index=dates)
                
                if len(series_annual) == 0:
                    print(f"  ✗ {filename}: No valid data after parsing")
                    return None
            
            # ===== DETECT & ADJUST FISCAL YEAR =====
            # Heuristic: if year values suggest FY (Oct-Sept), adjust to CY
            # Most federal spending data (Medicaid, SNAP, UI) reports on FY basis
            years = series_annual.index.year
            
            if is_fiscal_year and len(series_annual) > 0:
                # Shift FY dates to Jan 1 of calendar year
                adjusted_index = pd.to_datetime([f"{y}-01-01" for y in years])
                series_annual.index = adjusted_index
                adj_note = " (FY→CY adjusted)"
            else:
                adj_note = ""
            
            # ===== ALIGN TO QUARTERLY =====
            aligned = self._align_to_quarterly(
                series_annual, 
                series_name, 
                fill_method='ffill', 
                fill_gaps_with_zero=False
            )
            
            # Store in translated series
            self.translated_series.setdefault(country, {})[series_name] = aligned
            if series_name in self.diagnostics:
                self.diagnostics[series_name]['country'] = country
            
            print(f"  ✓ {filename} → {series_name} ({len(series_annual)} years{adj_note})")
            
            return aligned
        
        except Exception as e:
            print(f"  ✗ {filename}: {str(e)[:60]}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_diagnostics(self):
        """Print alignment diagnostics."""
        print("\n" + "="*80)
        print("DATA INGESTION DIAGNOSTICS")
        print("="*80)
        
        for series_name, diag in self.diagnostics.items():
            print(f"\n{series_name}:")
            print(f"  Original records: {diag['n_original']}")
            print(f"  Aligned quarters: {diag['n_aligned']}")
            print(f"  Coverage: {diag['coverage_pct']:.1f}%")
            print(f"  Date range: {diag['date_range']}")
        
        print("\n" + "="*80)

    def defaultIngestion(self, country = "United States"):
        self.ingest_and_adjust_fiscal_year(
            filename="gdp_nonUS.csv",
            series_name="gdp_billions", #not yet in billions needs to be translated..
            file_type='csv',
            is_fiscal_year=False,
            country=country,
            wide_year_columns=True,
            country_filter=country
        )
        self.translated_series[country]['gdp_billions'] = self.translated_series[country]['gdp_billions'] / 1e9
        self.ingest_and_adjust_fiscal_year(
            filename="political-violence.csv",
            date_col="Year",
            value_col="PTS_A",
            series_name="political_violence_score",
            file_type='csv',
            is_fiscal_year=False,
            country=country
        )
        self.ingest_and_adjust_fiscal_year(
            filename="swiid_gini.csv",
            date_col="year",
            value_col="gini_disp",
            series_name="gini_coefficient",
            file_type='csv',
            is_fiscal_year=False,
            country=country
        )
        self.ingest_and_adjust_fiscal_year(
            filename="tax_GDPpct_nonUS.csv",
            series_name="fed_tax_revenue_pct_gdp",
            file_type='csv',
            is_fiscal_year=False,
            country=country,
            wide_year_columns=True,
            country_filter=country
        )
        self.ingest_and_adjust_fiscal_year(
            filename="unemployment_nonUS.csv",
            series_name="unemployment_rate_pct",
            file_type='csv',
            is_fiscal_year=False,
            country=country,
            wide_year_columns=True,
            country_filter=country
        )

    
    def validate_alignment(self, reference_series_name=None, country="United States"):
        """Check for major misalignments between series, for one country."""
        print("\n" + "-"*80)
        print("ALIGNMENT VALIDATION")
        print("-"*80)
        
        series_dict = self.translated_series.get(country, {})
        if not series_dict:
            print(f"No series ingested yet for {country}.")
            return
        
        # Use first series as reference if not specified
        if reference_series_name is None:
            reference_series_name = list(series_dict.keys())[0]
        
        if reference_series_name not in series_dict:
            print(f"Reference series {reference_series_name} not found. Available:")
            for name in series_dict.keys():
                print(f"  - {name}")
            return
        
        ref_series = series_dict[reference_series_name]
        ref_coverage = ref_series.notna().sum()
        
        print(f"Reference series: {reference_series_name} ({ref_coverage} quarters)")
        
        for series_name, series in series_dict.items():
            if series_name == reference_series_name:
                continue
            
            coverage = series.notna().sum()
            
            # Check for misaligned dates
            ref_dates = ref_series.index[ref_series.notna()]
            series_dates = series.index[series.notna()]
            
            overlap = len(set(ref_dates) & set(series_dates))
            
            status = "✓" if overlap > 0 else "✗"
            print(f"{status} {series_name:30} | {coverage:3} qtrs | {overlap:3} overlap with ref")
        
        print("-"*80 + "\n")
    
    def merge_into_dataframe(self, existing_df, country="United States"):
        """Merge one country's translated series into existing DataFrame."""
        merged = existing_df.copy()
        
        for series_name, series in self.translated_series.get(country, {}).items():
            merged[series_name] = series
        
        return merged

    def compute_pct_gdp(self, country, numerator_series_name, output_series_name="redist_gdp_pct"):
        """
        Computes numerator / gdp_billions * 100 for one country using
        series already sitting in self.translated_series -- no separate
        GDP file needed, reuses whatever defaultIngestion() already
        pulled for that country.
        """
        cs = self.translated_series.get(country, {})
        if numerator_series_name not in cs or 'gdp_billions' not in cs:
            print(f"  ⚠ compute_pct_gdp({country}): missing '{numerator_series_name}' "
                  f"or 'gdp_billions' -- skipped, nothing fabricated.")
            return
        pct = cs[numerator_series_name] / cs['gdp_billions'] * 100
        self.translated_series[country][output_series_name] = pct
        print(f"  ✓ {country}: {output_series_name} computed in-line from "
              f"{numerator_series_name} / gdp_billions")


class USStateResponseDataCollector:
    """
    Collects and aggregates US economic data for state fiscal response calibration.
    
    State inputs (observables):
    - Unemployment rate
    - Gini coefficient (income inequality)
    - Wage growth (nominal, by percentile)
    - Job openings
    - Strike frequency / protest intensity
    
    State outputs (policy responses):
    - Redistribution spending (aggregated from multiple programs)
    - Tax rates (federal)
    - Social spending as % of GDP
    """
    
    def __init__(self, start_year=1960, end_year=2025, api_key=None):
        self.start_year = start_year
        self.end_year = end_year
        self.api_key = api_key
        
        if FRED_AVAILABLE and api_key:
            self.fred = Fred(api_key=api_key)
        else:
            self.fred = None
            print("⚠ FRED API unavailable. Will skip FRED data pulls.")
        
        # Main data container
        self.data = pd.DataFrame()
        
        # Redistribution program data (to be manually populated)
        self.redistribution_components = {}
        
        # Log
        self.log = []
    
    def _log(self, msg, level="INFO"):
        """Internal logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level:7} | {msg}")
        self.log.append((timestamp, level, msg))
    
    def fetch_fred_quarterly(self, series_id, column_name):
        """
        Fetch quarterly data from FRED. If monthly, average to quarterly.
        If annual, forward-fill to quarterly.
        """
        if not self.fred:
            self._log(f"Skipping {series_id} (FRED unavailable)", "WARN")
            return False
        
        try:
            series = self.fred.get_series(
                series_id,
                observation_start=f"{self.start_year}-01-01",
                observation_end=f"{self.end_year}-12-31"
            )
            
            # Convert to quarterly
            if len(series) > 0:
                # Resample: if monthly, take mean; if annual, forward-fill
                quarterly = series.resample('QE').mean()
                
                # If too sparse (annual data), forward-fill
                if quarterly.isnull().sum() / len(quarterly) > 0.5:
                    quarterly = series.resample('QE').ffill()
                
                self.data[column_name] = quarterly
                self._log(f"✓ {series_id} → {column_name} ({len(quarterly)} obs)")
                return True
            else:
                self._log(f"✗ {series_id}: No data returned", "WARN")
                return False
        
        except Exception as e:
            self._log(f"✗ {series_id}: {str(e)[:60]}", "ERROR")
            return False
    
    def add_manual_data(self, column_name, dates, values):
        """
        Manually add data series (for data you've collected externally).
        
        Args:
            column_name: str
            dates: list of dates or pd.DatetimeIndex
            values: list/array of values
        """
        try:
            series = pd.Series(values, index=pd.to_datetime(dates))
            self.data[column_name] = series
            self._log(f"✓ Added manual series: {column_name} ({len(series)} obs)")
        except Exception as e:
            self._log(f"✗ Failed to add {column_name}: {e}", "ERROR")
    
    def collect_labor_market_data(self):
        """Unemployment rate, participation, wages, job openings"""
        self._log("Collecting labor market data...")
        
        # Unemployment rate (quarterly average)
        self.fetch_fred_quarterly("UNRATE", "unemployment_rate_pct")
        
        # Labor force participation rate
        self.fetch_fred_quarterly("CIVPART", "lfpr_pct")
        
        # Average hourly earnings, private sector (nominal, not seasonally adjusted for consistency)
        self.fetch_fred_quarterly("CES0500000003", "avg_hourly_earnings_nominal")
        
        # Real average hourly earnings (inflation-adjusted)
        self.fetch_fred_quarterly("CES3000000008", "avg_hourly_earnings_real")
        
        # Job openings (JOLTS, starts 2000)
        self.fetch_fred_quarterly("JTSJOR", "job_openings_thousands")
        
        # Employment level
        self.fetch_fred_quarterly("PAYEMS", "total_nonfarm_employment")
        
        # Total civilian labor force (for strike participation rate normalization)
        self.fetch_fred_quarterly("CLF16OV", "civilian_labor_force_thousands")
        
        self._log("Labor market data collection complete")
    
    def collect_income_inequality_data(self):
        """
        Gini coefficient, income shares.
        
        NOTE: Gini is NOT on FRED. Must be manually collected from:
        - Census Bureau: https://www.census.gov/topics/income-poverty/income/data/tables.html
        - World Bank: https://data.worldbank.org/indicator/SI.POV.GINI
        - SWIID (Standardized World Income Inequality Database)
        
        Example: Create CSV with columns [date, gini] and use add_manual_data()
        """
        self._log("⚠ Gini coefficient: Manual collection required", "WARN")
        self._log("  Source: Census Bureau (annual) or World Bank (annual)")
        self._log("  After downloading, use add_manual_data('gini_coefficient', dates, values)")
    
    def collect_wage_percentile_data(self):
        """
        Wage inequality: P10, P50, P90 (10th, 50th, 90th percentiles).
        
        NOTE: Must be manually collected from:
        - Census Bureau Current Population Survey
        - Pew Research Center (pre-processed)
        - IPUMS-CPS (University of Minnesota)
        
        These typically have annual frequency only.
        
        Derived measures:
        - P90/P10 ratio (wage inequality)
        - P90/P50 ratio (top half inequality)
        - P50/P10 ratio (bottom half inequality)
        """
        self._log("Collecting wage percentiles from BLS via FRED...")
        
        # Wage percentiles: Lowest 20% (1st-20th percentile)
        self.fetch_fred_quarterly("CXU900000LB0102M", "wage_p10")
        
        # Wage percentiles: 4th quintile (50th-60th percentile proxy)
        self.fetch_fred_quarterly("CXU900000LB0103M", "wage_p50")
        
        # Wage percentiles: Highest 20% (80th-100th percentile)
        self.fetch_fred_quarterly("CXU900000LB0105M", "wage_p90")
        
        self._log("Wage percentile data collection complete")
    
    def collect_tax_data(self):
        """Federal tax revenue, effective tax rates"""
        self._log("Collecting tax data...")
        
        # Federal government current receipts (% of GDP)
        self.fetch_fred_quarterly("W068RCQ027SBEA", "fed_tax_revenue_pct_gdp")
        
        # Average federal income tax rate
        # NOTE: Not on FRED. Must manually collect from Tax Foundation:
        # https://taxfoundation.org/data/all/federal-tax-rates-history/
        self._log("⚠ Average federal income tax rate: Manual collection from Tax Foundation", "WARN")
        
        self._log("Tax data collection (partial)")
    
    def aggregate_redistribution(self):
        """
        Combine all redistribution components into single series.
        Fills NaN values where programs didn't have data for all years.
        """
        self._log("Aggregating redistribution programs...")
        
        redist_cols = [c for c in self.data.columns if c.startswith('redist_')]
        
        if not redist_cols:
            self._log("No redistribution components found. Check manual data uploads.", "WARN")
            return
        
        # Sum across programs (handles NaN by default)
        self.data['redistribution_total_billions'] = self.data[redist_cols].sum(axis=1, skipna=True)
        
        # Convert to % of GDP
        if 'gdp_billions' in self.data.columns:
            self.data['redistribution_pct_gdp'] = (
                self.data['redistribution_total_billions'] / 
                self.data['gdp_billions'] * 100
            )
        else:
            self._log("GDP data not available. Cannot compute redistribution % GDP", "WARN")
        
        self._log(f"✓ Aggregated redistribution from {len(redist_cols)} programs")
    
    def collect_gdp_data(self):
        """Nominal GDP (for % calculations)"""
        self._log("Collecting GDP data...")
        self.fetch_fred_quarterly("GDP", "gdp_billions")
    
    def collect_protest_data(self):
        """
        Strike frequency and protest intensity.
        
        Manual collection from:
        - Cross-National Time-Series (CNTSDA): Strike data 1960-2012
        - BLS (Bureau of Labor Statistics): Work stoppages
        - ACLED (Armed Conflict Location & Event Data): Protest events
        - Count of protests per quarter from ACLED dataset
        
        Available: https://acleddata.com/
        """
        self._log("⚠ Protest/strike data: Manual collection required", "WARN")
        self._log("  Source 1: BLS work stoppages (number and workers affected)")
        self._log("  Source 2: CNTSDA strike frequency (free download)")
        self._log("  Source 3: ACLED protest events (downloadable)")
        self._log("  Suggested: Count strikes per quarter, normalize by labor force")
    
    def clean_and_align(self):
        """
        Align all series to quarterly frequency, forward-fill gaps.
        """
        self._log("Cleaning and aligning data...")
        
        # Resample to quarterly, forward-fill annual data
        #prev, unworking --> self.data = self.data.resample('QE').interpolate(method='ffill')
        self.data = self.data.resample("QE").ffill()
        
        # Drop rows where all values are NaN
        self.data = self.data.dropna(how='all')
        
        self._log(f"✓ Data aligned to quarterly frequency ({len(self.data)} quarters)")
        self._log(f"  Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
    
    def calculate_derived_metrics(self):
        """
        Calculate metrics needed for state response fitting:
        - Unemployment rate (already have)
        - Wage growth (YoY)
        - Gini coefficient (already have from manual)
        - Redistribution level
        - Protest intensity (normalized)
        - Labor market tightness (job openings / unemployment)
        """
        self._log("Calculating derived metrics...")
        
        # Wage growth (year-over-year, quarterly data)
        if 'avg_hourly_earnings_nominal' in self.data.columns:
            self.data['wage_growth_yoy'] = self.data['avg_hourly_earnings_nominal'].pct_change(4) * 100
        
        # Labor market tightness (job openings / unemployment level, starts 2000)
        if 'job_openings_thousands' in self.data.columns and 'unemployment_rate_pct' in self.data.columns:
            self.data['labor_market_tightness'] = (
                self.data['job_openings_thousands'] / 
                (self.data['unemployment_rate_pct'] / 100 * self.data['total_nonfarm_employment'])
            )
        
        self._log("✓ Derived metrics calculated")
    
    def calculate_strike_severity(self, workers_affected_col, days_idle_col,
                                output_col='protest_intensity_score',
                                days_per_quarter=91):
        """
        Composite strike severity score, rebuilt to avoid saturation.

        Old version divided cumulative person-days-idle by 365 and clipped —
        since days_idle is a *summed* figure across all strikes in a quarter
        (often in the thousands+), that clip triggered almost every period,
        collapsing the score to a near-constant ~0.5. Fixed by:
        1. Normalizing days_idle against total *available* person-days
            (labor_force x days_per_quarter), not a flat 365.
        2. log1p-transforming both components before combining, since
            strike activity is extremely right-skewed (a few large strikes
            dominate raw counts).
        3. Min-max rescaling the combined score across the full series so
            it actually spans [0, 1] instead of saturating.
        """
        if 'civilian_labor_force_thousands' not in self.data.columns:
            self._log("Strike severity calc requires civilian_labor_force_thousands", "WARN")
            return

        if workers_affected_col not in self.data.columns or days_idle_col not in self.data.columns:
            self._log(f"Strike columns {workers_affected_col}, {days_idle_col} not found", "WARN")
            return

        labor_force = self.data['civilian_labor_force_thousands'] * 1000

        # Genuine rates (sanity-clipped only to guard against bad input data,
        # not expected to bind under normal values)
        participation_rate = (self.data[workers_affected_col] / labor_force).clip(0, 1)
        days_idle_rate = (self.data[days_idle_col] / (labor_force * days_per_quarter)).clip(0, 1)

        # Log-transform to tame right-skew (scale up first since rates are tiny,
        # e.g. ~1e-5, and log1p on values that small barely moves them)
        log_participation = np.log1p(participation_rate * 1e6)
        log_days_idle = np.log1p(days_idle_rate * 1e6)

        raw_score = log_participation + log_days_idle

        # Min-max rescale across the full series -> guarantees real spread in [0,1]
        valid = raw_score.dropna()
        if len(valid) == 0 or valid.max() == valid.min():
            self._log(f"{output_col}: insufficient variation to rescale", "WARN")
            self.data[output_col] = raw_score
            return

        self.data[output_col] = (raw_score - valid.min()) / (valid.max() - valid.min())

        self._log(f"✓ Strike severity recalculated → {output_col} "
                f"(range: {self.data[output_col].min():.3f}-{self.data[output_col].max():.3f})")

    def calculate_participation_rate(self, workers_affected_col,
                                       output_col='protest_participation_rate'):
        """
        Raw strike participation rate (workers_affected / labor_force),
        stored alongside protest_intensity_score rather than replacing it.

        protest_intensity_score is log-transformed and min-max rescaled
        across the series (see calculate_strike_severity) -- useful for
        the state's response-function fit, but its mean is an artifact of
        rescaling, not a real-world magnitude, so it isn't comparable to
        the ABM's simulated protest_share (a literal fraction of workers
        protesting per tick). This column is the actual comparable rate,
        left unscaled/untransformed so it can be used directly as a
        calibration target.
        """
        if 'civilian_labor_force_thousands' not in self.data.columns:
            self._log("Participation rate calc requires civilian_labor_force_thousands", "WARN")
            return

        if workers_affected_col not in self.data.columns:
            self._log(f"Strike column {workers_affected_col} not found", "WARN")
            return

        labor_force = self.data['civilian_labor_force_thousands'] * 1000

        self.data[output_col] = (
            self.data[workers_affected_col] / labor_force
        ).clip(0, 1)

        self._log(f"✓ Participation rate calculated → {output_col} "
                  f"(mean: {self.data[output_col].mean():.5f}, "
                  f"range: {self.data[output_col].min():.5f}-{self.data[output_col].max():.5f})")
    
    def validate_data(self):
        """Check for issues, report coverage"""
        self._log("Validating data quality...")
        
        print("\n" + "="*70)
        print("DATA COVERAGE REPORT")
        print("="*70)
        
        for col in self.data.columns:
            n_values = self.data[col].notna().sum()
            pct_coverage = (n_values / len(self.data)) * 100
            
            status = "✓" if pct_coverage > 80 else "⚠" if pct_coverage > 50 else "✗"
            print(f"{status} {col:40} | {pct_coverage:5.1f}% ({n_values:4} / {len(self.data)} quarters)")
        
        print("="*70 + "\n")
        
        self._log("✓ Validation complete")
    
    def export_data(self, filepath="results/us_state_response_data.csv"):
        """Save to CSV"""
        try:
            self.data.to_csv(filepath)
            self._log(f"✓ Data exported to {filepath}")
            return filepath
        except Exception as e:
            self._log(f"✗ Export failed: {e}", "ERROR")
            return None
    
    def run_collection(self):
        """Execute full data collection pipeline"""
        self._log("="*70)
        self._log("US STATE RESPONSE FUNCTION DATA COLLECTION (1960-2025)")
        self._log("="*70)
        
        self.collect_gdp_data()
        self.collect_labor_market_data()
        self.collect_income_inequality_data()
        self.collect_wage_percentile_data()
        self.collect_tax_data()
        self.collect_protest_data()
        
        self._log("\n" + "-"*70)
        self._log("MANUAL DATA COLLECTION CHECKLIST")
        self._log("-"*70)
        
        manual_items = [
            ("Gini coefficient (annual)", "Census Bureau / World Bank"),
            ("Federal income tax rate (annual)", "Tax Foundation"),
            ("SNAP benefits (annual)", "USDA / Census"),
            ("EITC claims (annual)", "IRS / Tax Foundation"),
            ("Strike/protest data (quarterly)", "BLS / CNTSDA / ACLED"),
        ]
        
        print("\nBefore running aggregation, manually collect:")
        for i, (item, source) in enumerate(manual_items, 1):
            print(f"  {i}. {item}")
            print(f"     → {source}")
        
        print("\nThen call:")
        print("  collector.add_manual_data(col_name, dates, values)")
        print("  collector.aggregate_redistribution()")
        print("  collector.clean_and_align()")
        print("  collector.calculate_derived_metrics()")
        print("  collector.validate_data()")
        print("  collector.export_data()\n")
        
        self._log("✓ Collection pipeline initialized. Await manual data uploads.")

def parse_ambiguous_strike_dates(date_series):
    """
    Parse strike dates with mixed MM/DD/YY and MM/DD/YYYY formats.
    
    Logic:
    - Extract year from MM/DD/YY or MM/DD/YYYY
    - If YY (2-digit): YY > 50 → 19YY, YY <= 50 → 20YY
    - If YYYY (4-digit): use as-is
    - Return only year (discard MM/DD)
    
    Args:
        date_series: pd.Series with dates like "2/16/88", "1/19/93", "2/24/2000"
    
    Returns:
        pd.Series with years as integers
    """
    years = []
    
    for date_str in date_series:
        if pd.isna(date_str):
            years.append(np.nan)
            continue
        
        date_str = str(date_str).strip()
        
        # Parse MM/DD/YY or MM/DD/YYYY
        match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', date_str)
        
        if match:
            year_str = match.group(3)
            
            if len(year_str) == 2:
                # Ambiguous YY format
                yy = int(year_str)
                if yy > 50:
                    year = 1900 + yy
                else:
                    year = 2000 + yy
            else:
                # 4-digit year
                year = int(year_str)
            
            years.append(year)
        else:
            # If parse fails, set NaN
            years.append(np.nan)
    
    return pd.Series(years, index=date_series.index)

def run_manual_ingestion(collector_data, data_dir="data/raw/"):
    """Run manual data ingestion with unified FY handling."""
    
    print("\n" + "="*80)
    print("MANUAL DATA INGESTION PIPELINE")
    print("="*80 + "\n")
    
    translator = ManualDataTranslator(
        data_dir=data_dir,
        quarterly_index=collector_data.index
    )
    
    print("Ingesting data files...\n")
    
    # Calendar year data (no FY adjustment needed)
    translator.ingest_and_adjust_fiscal_year(
        filename="gini.csv",
        date_col="reporting_year",
        value_col="gini",
        series_name="gini_coefficient",
        file_type='csv',
        is_fiscal_year=False  # ← Calendar year
    )
    
    translator.ingest_and_adjust_fiscal_year(
        filename="federal-tax.csv",
        date_col="Year",
        value_col="Total",
        series_name="avg_federal_tax_rate_pct",
        file_type='csv',
        skip_rows=1,
        is_fiscal_year=False  # ← Calendar year
    )
    
    translator.ingest_and_adjust_fiscal_year(
        filename="EITC_to_2025.xlsx",
        date_col="Year",
        value_col="total_eitc_billions",
        series_name="redist_eitc",
        file_type='xlsx',
        sheet_name="Sheet1",
        is_fiscal_year=False  # ← Calendar year
    )
    
    # Fiscal year data (needs FY→CY adjustment)
    translator.ingest_and_adjust_fiscal_year(
        filename="SNAP.xlsx",
        date_col="Fiscal_Year",
        value_col="snap_benefits_millions",
        series_name="redist_snap",
        file_type='xlsx',
        sheet_name="Sheet1",
        is_fiscal_year=True  # ← Fiscal year
    )
    
    translator.ingest_and_adjust_fiscal_year(
        filename="medicaid.csv",
        date_col="Year",
        value_col="expenditure_billions",
        series_name="redist_medicaid",
        file_type='csv',
        is_fiscal_year=True  # ← Fiscal year
    )
    
    # Calendar year (full date format)
    translator.ingest_and_adjust_fiscal_year(
        filename="UI.csv",
        date_col="observation_date",
        value_col="money_billions",
        series_name="redist_ui",
        file_type='csv',
        is_fiscal_year=False  # ← Calendar year
    )

    translator.ingest_and_merge_strike_data(
        annual_file="annual-strike-listing.xlsx",
        detailed_file="strike-listing.xlsx",
        annual_date_col="Year",
        annual_workers_col="num_workers",
        annual_days_col="idle_days",
        detailed_date_col="stoppage_date",
        detailed_workers_col="num_workers",
        detailed_days_col="idle_days",
        cutover_year=1988
    )

    translator.ingest_cntsdata_protests(
        filename="CNTSDATA.xlsx",
        sheet_name="2026 Data",
        year_col="Year",
        value_cols=["riots", "demonstrations", "strikes"],
        series_name="civil_unrest_events",
        country_filter="United States"
    )

    translator.defaultIngestion(country="Chile")
    translator.ingest_and_adjust_fiscal_year(
        filename="oecd.csv",
        date_col="Year",
        value_col="Public social expenditure as a share of GDP",
        series_name="redist_gdp_pct",
        file_type='csv',
        is_fiscal_year=False,
        country_col="Entity",
        country="Chile",
        country_filter="Chile"
    )
    translator.defaultIngestion(country="Germany")
    translator.ingest_and_adjust_fiscal_year(
        filename="oecd.csv",
        date_col="Year",
        value_col="Public social expenditure as a share of GDP",
        series_name="redist_gdp_pct",
        file_type='csv',
        is_fiscal_year=False,
        country_col="Entity",
        country="Germany",
        country_filter="Germany"
    )
    translator.defaultIngestion(country="Egypt")
    translator.ingest_and_adjust_fiscal_year(
        filename="welfare_expenditure.csv",
        date_col="Year",
        value_col="value",
        series_name="redist_usd_bn",
        file_type='csv',
        is_fiscal_year=False,
        country_col="Country",
        country="Egypt",
        country_filter="Egypt"
    )
    translator.compute_pct_gdp(country="Egypt", numerator_series_name="redist_usd_bn")
    translator.defaultIngestion(country="United Kingdom")
    translator.ingest_and_adjust_fiscal_year(
        filename="oecd.csv",
        date_col="Year",
        value_col="Public social expenditure as a share of GDP",
        series_name="redist_gdp_pct",
        file_type='csv',
        is_fiscal_year=False,
        country_col="Entity",
        country="United Kingdom",
        country_filter="United Kingdom"
    )
    translator.defaultIngestion(country="Russia")
    translator.ingest_and_adjust_fiscal_year(
        filename="welfare_expenditure.csv",
        date_col="Year",
        value_col="value",
        series_name="redist_gdp_pct",
        file_type='csv',
        is_fiscal_year=False,
        country_col="Country",
        country="Russia",
        country_filter="Russia"
    )
    translator.defaultIngestion(country="South Africa")
    translator.ingest_and_adjust_fiscal_year(
        filename="welfare_expenditure.csv",
        date_col="Year",
        value_col="value",
        series_name="redist_gdp_pct",
        file_type='csv',
        is_fiscal_year=False,
        country_col="Country",
        country="South Africa",
        country_filter="South Africa"
    )
    translator.defaultIngestion(country="Korea")
    translator.ingest_and_adjust_fiscal_year(
        filename="oecd.csv",
        date_col="Year",
        value_col="Public social expenditure as a share of GDP",
        series_name="redist_gdp_pct",
        file_type='csv',
        is_fiscal_year=False,
        country_col="Entity",
        country="Korea",
        country_filter="Korea"
    )
    translator.defaultIngestion(country="Poland")
    
    # ===== VALIDATION =====
    print("\n")
    translator.print_diagnostics()
    translator.validate_alignment(reference_series_name="gini_coefficient")
    
    # ===== MERGE =====
    print("\nMerging with FRED data...\n")
    merged_df = translator.merge_into_dataframe(collector_data)

    merged_df['redist_snap'] = merged_df['redist_snap'] / 1000
    merged_df['redist_eitc'] = merged_df['redist_eitc'] / 1000
    
    print(f"✓ Complete (all series on calendar year basis, quarterly aligned)")
    print(f"  Shape: {merged_df.shape}")
    print(f"  Date range: {merged_df.index.min().date()} to {merged_df.index.max().date()}")
    
    return merged_df, translator


# ============================================================================
# FINAL ASSEMBLY: long-format combine (country, year, label, value)
# ============================================================================
# Everything above this line is unchanged from the original US pipeline
# (only addition: ingest_and_adjust_fiscal_year now takes an optional
# `country` param, default "United States", so nothing existing breaks).
#
# Non-US countries are all manually collected -- no FRED/API calls for
# them anywhere below. This section just melts wide data (US quarterly,
# or a country's own wide/annual frame) into one tidy long table.
# ============================================================================

def melt_to_long_format(df, country, label_col_name='label'):
    """
    Melts a wide dataframe (columns = variables, index = date or year) into
    long format: country, year, label, value.

    If df's index looks quarterly (has a .quarter attribute via DatetimeIndex),
    collapses to annual (mean of the year's quarters) first, so US data ends
    up on the same annual grain as every manually-collected country -- no
    non-US source here is finer than annual anyway.

    Drops NaN rows (a missing quarter/year for a given label is just absent
    from the output, not filled or interpolated here).
    """
    d = df.copy()

    if isinstance(d.index, pd.DatetimeIndex):
        d['year'] = d.index.year
        d = d.groupby('year').mean(numeric_only=True).reset_index()
    elif 'year' not in d.columns:
        raise ValueError("df needs either a DatetimeIndex or a 'year' column")

    long_df = d.melt(id_vars='year', var_name=label_col_name, value_name='value')
    long_df = long_df.dropna(subset=['value'])
    long_df.insert(0, 'country', country)
    return long_df[['country', 'year', label_col_name, 'value']]


def load_manual_country_csv(filepath, country, year_col="Year", value_col="value",
                             country_col="Country", label="redistribution_pct_gdp",
                             min_year=None, max_year=None):
    """
    Reads a manually-maintained multi-country CSV (e.g.
    welfare_expenditures.csv), filters to one country, and returns rows
    already in the long format: country, year, label, value.

    Handles a raw value cell that's a plain number OR has a trailing
    currency-style 'bn' suffix (e.g. Egypt's "310.862bn") -- strips commas/%
    /'bn' before parsing. No FRED/API call, this is manual-data-only, and no
    interpolation across missing years -- a gap stays a gap.

    min_year/max_year: optional hard bounds, e.g. min_year=1992 for Russia
    to keep pre-1991 Soviet-era rows out even though the source CSV's
    country label ("Soviet Union / Russia") spans both eras.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  ✗ {filepath}: not found (manual data only, nothing auto-fetched)")
        return pd.DataFrame(columns=['country', 'year', 'label', 'value'])

    df = pd.read_csv(filepath)
    df.columns = [str(c).strip() for c in df.columns]
    sub = df[df[country_col] == country].copy()

    def _parse(raw):
        if pd.isna(raw):
            return np.nan
        s = str(raw).strip().replace(',', '').replace('%', '')
        s = re.sub(r'bn$', '', s, flags=re.IGNORECASE).strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    sub['year'] = pd.to_numeric(sub[year_col], errors='coerce')
    sub['value'] = sub[value_col].apply(_parse)
    sub = sub.dropna(subset=['year', 'value'])
    if min_year is not None:
        sub = sub[sub['year'] >= min_year]
    if max_year is not None:
        sub = sub[sub['year'] <= max_year]

    sub['year'] = sub['year'].astype(int)
    sub['label'] = label
    sub['country'] = country
    print(f"  ✓ {filepath.name} ({country}) → {len(sub)} year(s)")
    return sub[['country', 'year', 'label', 'value']]


def convert_egypt_to_pct_gdp(egypt_usd_bn_long, gdp_usd_csv):
    """
    Egypt-specific in-line %GDP calc -- the one genuinely different
    operation among the non-US countries (FX-converted USD spending /
    Egypt GDP in current USD), so it stays a separate small function
    rather than a parameter to the generic loader above.

    egypt_usd_bn_long: long-format df (country/year/label/value) of
    Egypt's redistribution in USD billions, e.g. from
    load_manual_country_csv(..., label='redistribution_usd_bn').
    gdp_usd_csv: path to a simple Year,value CSV of Egypt GDP in current
    USD (billions). If not found, returns an empty frame rather than
    fabricating a %GDP figure.
    """
    gdp_path = Path(gdp_usd_csv)
    if not gdp_path.exists():
        print(f"  ⚠ {gdp_path}: not found -- Egypt redistribution_pct_gdp "
              f"skipped, not fabricated. Add this file (Year, value in "
              f"current USD) and re-run.")
        return pd.DataFrame(columns=['country', 'year', 'label', 'value'])

    gdp = pd.read_csv(gdp_path)
    gdp.columns = [str(c).strip() for c in gdp.columns]
    gdp_map = dict(zip(pd.to_numeric(gdp['Year'], errors='coerce'),
                        pd.to_numeric(gdp['value'], errors='coerce')))

    out = egypt_usd_bn_long.copy()
    out['value'] = out.apply(
        lambda r: (r['value'] / gdp_map[r['year']] * 100)
        if r['year'] in gdp_map and gdp_map[r['year']] else np.nan,
        axis=1
    )
    out['label'] = 'redistribution_pct_gdp'
    out = out.dropna(subset=['value'])
    print(f"  ✓ Egypt redistribution_pct_gdp computed in-line for {len(out)} year(s)")
    return out


if __name__ == "__main__":

    collector = USStateResponseDataCollector(
        start_year=1960,
        end_year=2025,
        api_key=FRED_API_KEY
    )

    collector.run_collection()
    collector.clean_and_align()
    collector.calculate_derived_metrics()

    merged, translator = run_manual_ingestion(collector.data, data_dir="data/raw/")
    collector.data = merged

    if 'workers_affected' in merged.columns:
        collector.calculate_strike_severity('workers_affected', 'days_idle')
        collector.calculate_participation_rate('workers_affected')

    collector.validate_data()
    collector.export_data("results/us_state_response_data.csv")

    # ===== FINAL ASSEMBLY: long-format combine =====
    print("\n" + "=" * 70)
    print("FINAL LONG-FORMAT ASSEMBLY (country, year, label, value)")
    print("=" * 70)

    us_long = melt_to_long_format(collector.data, country="United States")

    # Every non-US country's data now lives in translator.translated_series,
    # keyed by country -- build one wide frame per country and melt it,
    # instead of re-reading CSVs a second time through a separate path.
    combined_frames = [us_long]
    for country, series_dict in translator.translated_series.items():
        if country == "United States":
            continue  # already in us_long via collector.data above
        if not series_dict:
            continue  # e.g. Poland -- deliberately no numeric series
        country_df = pd.DataFrame(series_dict)
        combined_frames.append(melt_to_long_format(country_df, country=country))

    combined_long = pd.concat(combined_frames, ignore_index=True)

    combined_long = combined_long.sort_values(['country', 'label', 'year']).reset_index(drop=True)
    combined_long.to_csv("results/combined_long_panel.csv", index=False)

    print(f"\n  Countries: {sorted(combined_long['country'].unique())}")
    print(f"  Labels: {sorted(combined_long['label'].unique())}")
    print(f"  Rows: {len(combined_long)}")
    print(f"  Saved to: results/combined_long_panel.csv")