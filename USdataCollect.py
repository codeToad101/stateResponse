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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    print("⚠ fredapi not installed. Install with: pip install fredapi")
    FRED_AVAILABLE = False


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
                quarterly = series.resample('Q').mean()
                
                # If too sparse (annual data), forward-fill
                if quarterly.isnull().sum() / len(quarterly) > 0.5:
                    quarterly = series.resample('Q').ffill()
                
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
        self._log("⚠ Wage percentiles (P10, P50, P90): Manual collection required", "WARN")
        self._log("  Source: Census Bureau CPS / Pew Research")
        self._log("  Frequency: Annual (resample to quarterly with forward-fill)")
        self._log("  After downloading, use add_manual_data() for each percentile")
    
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
    
    def collect_redistribution_data(self):
        """
        MANUAL AGGREGATION of redistribution programs:
        1. Unemployment Insurance (UI)
        2. Workers Compensation
        3. Veterans Benefits
        4. SNAP (Food Stamps)
        5. EITC (Earned Income Tax Credit)
        6. Medicaid (medical vendor payments)
        
        These will be manually collected and summed.
        """
        self._log("Setting up redistribution program components...")
        
        # FRED IDs for individual programs (to be fetched or manual)
        programs = {
            'unemployment_insurance': {
                'fred_id': 'IUPBS',  # Unemployment Insurance Benefits Paid
                'description': 'Quarterly benefits paid ($ billions)',
                'notes': 'Direct from FRED'
            },
            'workers_compensation': {
                'fred_id': 'WCOMPIB',  # Workers Compensation Insurance Benefits
                'description': 'Quarterly benefits paid ($ billions)',
                'notes': 'Direct from FRED'
            },
            'veterans_benefits': {
                'fred_id': 'VETERANS',  # Veterans Compensation & Pension
                'description': 'Quarterly benefits paid ($ billions)',
                'notes': 'Direct from FRED'
            },
            'snap': {
                'fred_id': 'A627RC1',  # SNAP Benefits Paid
                'description': 'Annual benefits paid ($ billions) - forward-fill to quarterly',
                'notes': 'Manual collection from USDA or Census'
            },
            'eitc': {
                'fred_id': None,
                'description': 'Annual tax credits claimed ($ billions)',
                'notes': 'Manual from IRS / Tax Foundation'
            },
            'medicaid': {
                'fred_id': 'A091MD3A027NBEA',  # Medicaid benefit payments
                'description': 'Quarterly benefit payments ($ billions)',
                'notes': 'Direct from FRED'
            }
        }
        
        for program_name, metadata in programs.items():
            if metadata['fred_id']:
                self.fetch_fred_quarterly(
                    metadata['fred_id'],
                    f"redist_{program_name}"
                )
            else:
                self._log(
                    f"⚠ {program_name.upper()}: Manual collection required",
                    "WARN"
                )
            
            self.redistribution_components[program_name] = metadata
        
        self._log("Redistribution programs: Ready for aggregation")
    
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
        self.data = self.data.resample('Q').interpolate(method='ffill')
        
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
        self.collect_redistribution_data()
        self.collect_protest_data()
        
        self._log("\n" + "-"*70)
        self._log("MANUAL DATA COLLECTION CHECKLIST")
        self._log("-"*70)
        
        manual_items = [
            ("Gini coefficient (annual)", "Census Bureau / World Bank"),
            ("Wage percentiles P10, P50, P90 (annual)", "Census CPS / Pew Research"),
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


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize (requires FRED API key from https://fred.stlouisfed.org)
    # API_KEY = "your_fred_api_key_here"
    
    collector = USStateResponseDataCollector(
        start_year=1960,
        end_year=2025,
        api_key=None  # Replace with your FRED API key
    )
    
    # Run automated collection (pulls what's available from FRED)
    collector.run_collection()
    
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