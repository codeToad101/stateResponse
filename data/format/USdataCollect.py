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
from secret import FRED_API_KEY
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
            
            self.translated_series['workers_affected'] = workers_aligned
            self.translated_series['days_idle'] = days_aligned
            
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
            self.translated_series[series_name] = aligned
            
            print(f"  ✓ {filename} (US only) → {series_name} ({len(series)} years)")
            return aligned
        
        except Exception as e:
            print(f"  ✗ {filename}: {str(e)[:50]}")
            return None

    def ingest_and_adjust_fiscal_year(self, filename, date_col, value_col, 
                                    series_name, file_type='xlsx', 
                                    sheet_name=None, encoding='utf-8', skip_rows=0,
                                    is_fiscal_year=True):
        """
        Ingest annual data (fiscal or calendar year), adjust if FY, then align to quarterly.
        Single flow with consistent diagnostics from _align_to_quarterly().
        
        Args:
            filename: str
            date_col: str, column name with dates
            value_col: str or list, column(s) with values
            series_name: str
            file_type: 'xlsx' or 'csv'
            sheet_name: str (only if xlsx)
            encoding: str (only if csv)
            skip_rows: int (only if csv)
        
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
            
            # Parse values (handle % and commas)
            if isinstance(value_col, list):
                data_values = df[value_col].sum(axis=1, skipna=True).values
            else:
                values_raw = df[value_col].astype(str).str.replace('%', '').str.replace(',', '')
                data_values = pd.to_numeric(values_raw, errors='coerce').values
            
            # Parse dates
            dates = pd.to_datetime(df[date_col], errors='coerce')
            
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
            
            # Simple heuristic: assume annual data ending in Oct-Sept is FY
            # (most common for US federal fiscal year programs)
            # If your data is explicitly labeled FY, set this flag manually
            is_fiscal_year = True  # Set to False if you KNOW it's calendar year
            
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
            self.translated_series[series_name] = aligned
            
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
    
    def validate_alignment(self, reference_series_name=None):
        """Check for major misalignments between series."""
        print("\n" + "-"*80)
        print("ALIGNMENT VALIDATION")
        print("-"*80)
        
        if not self.translated_series:
            print("No series ingested yet.")
            return
        
        # Use first series as reference if not specified
        if reference_series_name is None:
            reference_series_name = list(self.translated_series.keys())[0]
        
        if reference_series_name not in self.translated_series:
            print(f"Reference series {reference_series_name} not found. Available:")
            for name in self.translated_series.keys():
                print(f"  - {name}")
            return
        
        ref_series = self.translated_series[reference_series_name]
        ref_coverage = ref_series.notna().sum()
        
        print(f"Reference series: {reference_series_name} ({ref_coverage} quarters)")
        
        for series_name, series in self.translated_series.items():
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
    
    def merge_into_dataframe(self, existing_df):
        """Merge all translated series into existing DataFrame."""
        merged = existing_df.copy()
        
        for series_name, series in self.translated_series.items():
            merged[series_name] = series
        
        return merged


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
        self.fetch_fred_quarterly("A191RL1Q225SBEA", "gdp_billions")
    
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
    
    def calculate_strike_severity(self, workers_affected_col, days_idle_col, output_col='protest_intensity_score'):
        """
        Calculate composite strike severity score for state response function.
        
        Used after manual strike data ingestion. Combines:
        - Participation rate: (workers_affected / civilian_labor_force)
        - Duration: days_idle (cumulative disruption)
        
        Args:
            workers_affected_col: column name with workers affected
            days_idle_col: column name with cumulative days idle
            output_col: name for output severity score
        
        Returns:
            None (modifies self.data in place)
        """
        if 'civilian_labor_force_thousands' not in self.data.columns:
            self._log("Strike severity calc requires civilian_labor_force_thousands", "WARN")
            return
        
        if workers_affected_col not in self.data.columns or days_idle_col not in self.data.columns:
            self._log(f"Strike columns {workers_affected_col}, {days_idle_col} not found", "WARN")
            return
        
        # Participation rate (affected / total labor force, scaled to 0-1)
        participation = self.data[workers_affected_col] / (self.data['civilian_labor_force_thousands'] * 1000)
        participation = participation.clip(0, 1)  # cap at 100%
        
        # Days idle (normalize to 0-1 scale: cap at 365 days/quarter)
        days_scaled = self.data[days_idle_col] / 365
        days_scaled = days_scaled.clip(0, 1)
        
        # Composite score: equal weight to participation + duration
        self.data[output_col] = (participation + days_scaled) / 2
        
        self._log(f"✓ Strike severity calculated → {output_col}")
    
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
    
    def export_data(self, filepath="us_state_response_data.csv"):
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
    
    return merged_df

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    collector = USStateResponseDataCollector(
        start_year=1960,
        end_year=2025,
        api_key=FRED_API_KEY
    )
    
    # Run automated collection (pulls what's available from FRED)
    collector.run_collection()

    # Step 2: Clean & align
    collector.clean_and_align()
    collector.calculate_derived_metrics()
    collector.validate_data()

    # Step 3: Ingest manual data
    merged = run_manual_ingestion(collector.data, data_dir="data/raw/")
    collector.data = merged

    # Step 4: Calculate strike severity
    if 'workers_affected' in merged.columns:
        collector.calculate_strike_severity('workers_affected', 'days_idle')

    # Step 5: Export & inspect
    collector.export_data("us_state_response_data.csv")

    # Check shape & coverage
    print(f"\n✓ Final shape: {merged.shape}")
    print(f"\nColumn coverage:")
    print(merged.notna().sum() / len(merged) * 100)
    
    # ===== MANUAL DATA UPLOAD SECTION (Once you've collected external data) =====
    # 
    # Example: Add Gini data manually
    # gini_dates = pd.date_range('1960-01', '2025-12', freq='Y')  # Annual
    # gini_values = [0.394, 0.397, ...]  # Your data
    # collector.add_manual_data('gini_coefficient', gini_dates, gini_values)
    # 
    # Example: Add wage percentiles
    # collector.add_manual_data('wage_p10', dates_annual, values_p10)
    # collector.add_manual_data('wage_p50', dates_annual, values_p50)
    # collector.add_manual_data('wage_p90', dates_annual, values_p90)
    # 
    # Example: Add protest data
    # collector.add_manual_data('strikes_per_quarter', dates_quarterly, strike_counts)
    # 
    # ===== THEN RUN AGGREGATION =====
    # collector.aggregate_redistribution()
    # collector.clean_and_align()
    # collector.calculate_derived_metrics()
    # collector.validate_data()
    # collector.export_data("us_state_response_data.csv")
    # 
    # print("\n✓ Ready for state response function fitting!")
    # print(f"  Shape: {collector.data.shape}")
    # print(f"\nFirst few rows:")
    # print(collector.data.head())