# enhanced_data_quality_checker.py
"""
Enhanced Universal Data Quality Checker
Comprehensive checks for data quality issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

class EnhancedDataQualityChecker:
    """
    Comprehensive data quality checker with 15+ different checks
    """
    
    def __init__(self, df):
        self.df = df
        self.report = {
            'summary': {},
            'issues': [],
            'recommendations': [],
            'data_patterns': {}
        }
    
    def check_all(self):
        """Run all quality checks"""
        print("🔍 COMPREHENSIVE DATA QUALITY CHECK STARTING...")
        print("="*60)
        
        # Basic info
        self._check_basic_info()
        
        # Core quality checks
        self._check_missing_values()
        self._check_duplicates()
        self._check_data_types()
        self._check_outliers()
        self._check_unique_values()
        
        # Additional quality checks
        self._check_inconsistent_formatting()
        self._check_invalid_values()
        self._check_correlations()
        self._check_data_freshness()
        self._check_string_patterns()
        self._check_numeric_ranges()
        self._check_special_characters()
        self._check_data_consistency()
        self._check_data_completeness_patterns()
        self._check_potential_pii()
        
        # Create comprehensive visualizations
        self._create_enhanced_visualizations()
        
        # Generate detailed report
        self._print_detailed_report()
        
        return self.report
    
    def _check_basic_info(self):
        """Get basic dataset information"""
        self.report['summary'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'column_names': list(self.df.columns),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'text_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
    
    def _check_missing_values(self):
        """Check for missing values"""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        
        for col in self.df.columns:
            if missing[col] > 0:
                issue = {
                    'type': 'MISSING_VALUES',
                    'column': col,
                    'count': int(missing[col]),
                    'percentage': round(float(missing_percent[col]), 2),
                    'severity': 'HIGH' if missing_percent[col] > 50 else 'MEDIUM' if missing_percent[col] > 20 else 'LOW'
                }
                self.report['issues'].append(issue)
                
                # Add recommendation
                if missing_percent[col] > 90:
                    self.report['recommendations'].append(
                        f"❗ Column '{col}' has {missing_percent[col]:.1f}% missing values. Consider dropping this column."
                    )
                elif missing_percent[col] > 50:
                    self.report['recommendations'].append(
                        f"⚠️  Column '{col}' has {missing_percent[col]:.1f}% missing values. Handle with care."
                    )
    
    def _check_duplicates(self):
        """Check for duplicate rows"""
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            issue = {
                'type': 'DUPLICATE_ROWS',
                'count': int(dup_count),
                'percentage': round(float(dup_count / len(self.df) * 100), 2),
                'severity': 'HIGH' if dup_count > len(self.df) * 0.1 else 'MEDIUM'
            }
            self.report['issues'].append(issue)
            self.report['recommendations'].append(
                f"🔄 Found {dup_count} duplicate rows ({dup_count/len(self.df)*100:.1f}%). Consider removing them."
            )
    
    def _check_data_types(self):
        """Check for potential data type issues"""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Sample non-null values
                sample = self.df[col].dropna().astype(str)
                if len(sample) > 0:
                    # Check if it could be numeric
                    try:
                        pd.to_numeric(sample.head(100))
                        numeric_convertible = True
                    except:
                        numeric_convertible = False
                    
                    if numeric_convertible:
                        issue = {
                            'type': 'WRONG_DATA_TYPE',
                            'column': col,
                            'current_type': 'text',
                            'suggested_type': 'numeric',
                            'severity': 'LOW'
                        }
                        self.report['issues'].append(issue)
                        self.report['recommendations'].append(
                            f"💡 Column '{col}' looks numeric but is stored as text. Consider converting."
                        )
    
    def _check_outliers(self):
        """Check for outliers in numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 3:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                
                if outliers > 0:
                    issue = {
                        'type': 'OUTLIERS',
                        'column': col,
                        'count': int(outliers),
                        'percentage': round(float(outliers / len(data) * 100), 2),
                        'min_value': round(float(data.min()), 2),
                        'max_value': round(float(data.max()), 2),
                        'severity': 'MEDIUM'
                    }
                    self.report['issues'].append(issue)
                    
                    if outliers > len(data) * 0.05:  # More than 5% outliers
                        self.report['recommendations'].append(
                            f"📊 Column '{col}' has {outliers} outliers ({outliers/len(data)*100:.1f}%). Review these extreme values."
                        )
    
    def _check_unique_values(self):
        """Check for columns with too many or too few unique values"""
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_ratio = unique_count / len(self.df)
            
            # Constant columns (only 1 unique value)
            if unique_count == 1:
                issue = {
                    'type': 'CONSTANT_COLUMN',
                    'column': col,
                    'unique_values': 1,
                    'severity': 'LOW'
                }
                self.report['issues'].append(issue)
                self.report['recommendations'].append(
                    f"🔒 Column '{col}' has only 1 unique value. It provides no information."
                )
            
            # High cardinality warning
            elif self.df[col].dtype == 'object' and unique_ratio > 0.9:
                issue = {
                    'type': 'HIGH_CARDINALITY',
                    'column': col,
                    'unique_values': int(unique_count),
                    'unique_ratio': round(float(unique_ratio), 2),
                    'severity': 'LOW'
                }
                self.report['issues'].append(issue)
                self.report['recommendations'].append(
                    f"🏷️  Column '{col}' has {unique_count} unique values (almost all different). Might be an ID column."
                )
    
    def _check_inconsistent_formatting(self):
        """Check for inconsistent formatting in text columns"""
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].notna().any():
                sample = self.df[col].dropna()
                
                # Check for mixed case
                has_upper = sample.str.contains('[A-Z]', na=False).any()
                has_lower = sample.str.contains('[a-z]', na=False).any()
                all_upper = sample.str.isupper().all()
                all_lower = sample.str.islower().all()
                
                if has_upper and has_lower and not all_upper and not all_lower:
                    unique_cases = sample.str.lower().nunique()
                    original_unique = sample.nunique()
                    
                    if unique_cases < original_unique * 0.9:
                        issue = {
                            'type': 'INCONSISTENT_CASE',
                            'column': col,
                            'details': f'{original_unique - unique_cases} case variations',
                            'severity': 'LOW'
                        }
                        self.report['issues'].append(issue)
                
                # Check for leading/trailing spaces
                spaces = (sample.str.len() != sample.str.strip().str.len()).sum()
                if spaces > 0:
                    issue = {
                        'type': 'WHITESPACE_ISSUES',
                        'column': col,
                        'count': int(spaces),
                        'severity': 'LOW'
                    }
                    self.report['issues'].append(issue)
                    self.report['recommendations'].append(
                        f"🧹 Column '{col}' has {spaces} values with extra whitespace."
                    )
    
    def _check_invalid_values(self):
        """Check for invalid values in columns"""
        # Check for negative values where they shouldn't be
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if any(word in col.lower() for word in ['age', 'price', 'cost', 'count', 'quantity', 'year']):
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    issue = {
                        'type': 'INVALID_NEGATIVE_VALUES',
                        'column': col,
                        'count': int(negative_count),
                        'severity': 'MEDIUM'
                    }
                    self.report['issues'].append(issue)
                    self.report['recommendations'].append(
                        f"⚠️ Column '{col}' has {negative_count} negative values - this might be invalid."
                    )
        
        # Check for future dates
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    dates = pd.to_datetime(self.df[col], errors='coerce')
                    future_dates = (dates > datetime.now()).sum()
                    if future_dates > 0:
                        issue = {
                            'type': 'FUTURE_DATES',
                            'column': col,
                            'count': int(future_dates),
                            'severity': 'MEDIUM'
                        }
                        self.report['issues'].append(issue)
                except:
                    pass
    
    def _check_correlations(self):
        """Check for highly correlated columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            high_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr.append({
                            'columns': f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}",
                            'correlation': round(corr_matrix.iloc[i, j], 3)
                        })
            
            if high_corr:
                self.report['data_patterns']['high_correlations'] = high_corr
                self.report['recommendations'].append(
                    f"🔗 Found {len(high_corr)} pairs of highly correlated columns (>0.95). Consider removing redundancy."
                )
    
    def _check_data_freshness(self):
        """Check how recent the data is"""
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                try:
                    # Special handling for year columns
                    if 'year' in col.lower() and self.df[col].dtype in [int, float]:
                        years = self.df[col].dropna()
                        if years.max() < 2100 and years.min() > 1900:  # Reasonable year range
                            date_cols.append({
                                'column': col,
                                'latest': int(years.max()),
                                'oldest': int(years.min()),
                                'years_old': int(datetime.now().year - years.max())
                            })
                    else:
                        dates = pd.to_datetime(self.df[col], errors='coerce')
                        if dates.notna().any():
                            date_cols.append({
                                'column': col,
                                'latest': dates.max(),
                                'oldest': dates.min(),
                                'days_old': (datetime.now() - dates.max()).days
                            })
                except:
                    pass
        
        if date_cols:
            self.report['data_patterns']['date_ranges'] = date_cols
            for date_info in date_cols:
                if 'years_old' in date_info and date_info['years_old'] > 5:
                    self.report['recommendations'].append(
                        f"📅 Column '{date_info['column']}' latest year is {date_info['latest']} ({date_info['years_old']} years old). Data might be stale."
                    )
                elif 'days_old' in date_info and date_info['days_old'] > 365:
                    self.report['recommendations'].append(
                        f"📅 Column '{date_info['column']}' latest date is {date_info['days_old']} days old. Data might be stale."
                    )
    
    def _check_string_patterns(self):
        """Check for patterns in string columns"""
        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna()
            if len(sample) > 0:
                # Check for email patterns
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                potential_emails = sample.str.match(email_pattern).sum()
                
                # Check for phone patterns
                phone_pattern = r'^\+?[\d\s\-\(\)]+$'
                potential_phones = sample.str.match(phone_pattern).sum()
                
                # Check for URL patterns
                url_pattern = r'^https?://'
                potential_urls = sample.str.contains(url_pattern, na=False).sum()
                
                if potential_emails > len(sample) * 0.5:
                    self.report['data_patterns'][f'{col}_type'] = 'likely_email'
                elif potential_phones > len(sample) * 0.5:
                    self.report['data_patterns'][f'{col}_type'] = 'likely_phone'
                elif potential_urls > len(sample) * 0.5:
                    self.report['data_patterns'][f'{col}_type'] = 'likely_url'
    
    def _check_numeric_ranges(self):
        """Check if numeric values are within reasonable ranges"""
        for col in self.df.select_dtypes(include=[np.number]).columns:
            data = self.df[col].dropna()
            if len(data) > 0:
                # Check for suspiciously narrow ranges
                value_range = data.max() - data.min()
                if value_range == 0:
                    continue  # Already caught by constant check
                
                # Check for percentage columns (0-100)
                if 'percent' in col.lower() or '%' in col:
                    out_of_range = ((data < 0) | (data > 100)).sum()
                    if out_of_range > 0:
                        issue = {
                            'type': 'PERCENTAGE_OUT_OF_RANGE',
                            'column': col,
                            'count': int(out_of_range),
                            'severity': 'MEDIUM'
                        }
                        self.report['issues'].append(issue)
    
    def _check_special_characters(self):
        """Check for special characters that might cause issues"""
        special_chars_pattern = r'[^\w\s\-\.\,\@]'
        
        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna()
            if len(sample) > 0:
                has_special = sample.str.contains(special_chars_pattern, na=False).sum()
                if has_special > 0:
                    examples = sample[sample.str.contains(special_chars_pattern, na=False)].head(3).tolist()
                    issue = {
                        'type': 'SPECIAL_CHARACTERS',
                        'column': col,
                        'count': int(has_special),
                        'examples': examples[:3] if len(examples) > 0 else [],
                        'severity': 'LOW'
                    }
                    self.report['issues'].append(issue)
    
    def _check_data_consistency(self):
        """Check for logical consistency between columns"""
        # Example: Check if end dates are after start dates
        date_pairs = []
        for col1 in self.df.columns:
            if 'start' in col1.lower() and 'date' in col1.lower():
                for col2 in self.df.columns:
                    if 'end' in col2.lower() and 'date' in col2.lower():
                        date_pairs.append((col1, col2))
        
        for start_col, end_col in date_pairs:
            try:
                start_dates = pd.to_datetime(self.df[start_col], errors='coerce')
                end_dates = pd.to_datetime(self.df[end_col], errors='coerce')
                
                invalid = (start_dates > end_dates).sum()
                if invalid > 0:
                    issue = {
                        'type': 'INCONSISTENT_DATES',
                        'columns': f'{start_col} > {end_col}',
                        'count': int(invalid),
                        'severity': 'HIGH'
                    }
                    self.report['issues'].append(issue)
            except:
                pass
    
    def _check_data_completeness_patterns(self):
        """Check patterns of missing data"""
        missing_pattern = self.df.isnull()
        
        # Check if missing values are concentrated in certain rows
        rows_with_many_missing = (missing_pattern.sum(axis=1) > len(self.df.columns) * 0.5).sum()
        if rows_with_many_missing > 0:
            self.report['recommendations'].append(
                f"🚫 {rows_with_many_missing} rows have >50% missing values. Consider removing these rows."
            )
        
        # Check if certain columns are always missing together
        if missing_pattern.any().sum() > 1:
            missing_corr = missing_pattern.corr()
            for i in range(len(missing_corr.columns)):
                for j in range(i):
                    if abs(missing_corr.iloc[i, j]) > 0.9:
                        self.report['data_patterns']['missing_together'] = {
                            'columns': f"{missing_corr.columns[i]} & {missing_corr.columns[j]}",
                            'correlation': round(missing_corr.iloc[i, j], 2)
                        }
    
    def _check_potential_pii(self):
        """Check for potential personally identifiable information"""
        pii_keywords = ['ssn', 'social', 'email', 'phone', 'address', 'name', 
                       'birth', 'dob', 'passport', 'license', 'credit']
        
        potential_pii = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in pii_keywords):
                potential_pii.append(col)
        
        if potential_pii:
            self.report['data_patterns']['potential_pii'] = potential_pii
            self.report['recommendations'].append(
                f"🔐 Found {len(potential_pii)} columns that might contain PII. Ensure proper data protection."
            )
    
    def _create_enhanced_visualizations(self):
        """Create comprehensive visualizations"""
        try:
            fig = plt.figure(figsize=(16, 12))
            
            # Create 6 subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Missing values by column
            ax1 = fig.add_subplot(gs[0, :2])
            missing = self.df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=True)[:15]
            if len(missing) > 0:
                missing.plot(kind='barh', ax=ax1, color='coral')
                ax1.set_title('Top 15 Columns with Missing Values', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Number of Missing Values')
            
            # 2. Data completeness by row
            ax2 = fig.add_subplot(gs[0, 2])
            row_completeness = (self.df.notna().sum(axis=1) / len(self.df.columns)) * 100
            ax2.hist(row_completeness, bins=20, color='skyblue', edgecolor='black')
            ax2.set_title('Row Completeness Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Completeness %')
            ax2.set_ylabel('Number of Rows')
            
            # 3. Numeric distributions
            ax3 = fig.add_subplot(gs[1, :])
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]
            if len(numeric_cols) > 0:
                for i, col in enumerate(numeric_cols):
                    data = self.df[col].dropna()
                    if len(data) > 0:
                        ax3.hist(data, bins=30, alpha=0.5, label=col[:20])
                ax3.set_title('Numeric Column Distributions (First 4)', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.set_ylabel('Frequency')
            
            # 4. Correlation heatmap
            ax4 = fig.add_subplot(gs[2, :2])
            numeric_df = self.df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                # Limit to first 10 columns for readability
                if len(corr) > 10:
                    corr = corr.iloc[:10, :10]
                sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax4, 
                           cbar_kws={'label': 'Correlation'})
                ax4.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')
            
            # 5. Issue severity breakdown
            ax5 = fig.add_subplot(gs[2, 2])
            severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for issue in self.report['issues']:
                if 'severity' in issue:
                    severity_counts[issue['severity']] += 1
            
            if sum(severity_counts.values()) > 0:
                colors = ['#ff4444', '#ff8844', '#ffaa44']
                ax5.pie(severity_counts.values(), labels=severity_counts.keys(), 
                       colors=colors, autopct='%1.0f%%')
                ax5.set_title('Issues by Severity', fontsize=12, fontweight='bold')
            
            plt.suptitle('Enhanced Data Quality Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Warning: Could not create all visualizations: {str(e)}")
    
    def _print_detailed_report(self):
        """Print comprehensive report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE DATA QUALITY REPORT")
        print("="*60)
        
        # Summary
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Rows: {self.report['summary']['total_rows']:,}")
        print(f"   • Columns: {self.report['summary']['total_columns']}")
        print(f"   • Memory: {self.report['summary']['memory_usage_mb']:.1f} MB")
        print(f"   • Numeric columns: {self.report['summary']['numeric_columns']}")
        print(f"   • Text columns: {self.report['summary']['text_columns']}")
        
        # Quality score
        total_possible_issues = len(self.df.columns) * 10  # Rough estimate
        actual_issues = len(self.report['issues'])
        quality_score = max(0, 100 - (actual_issues / total_possible_issues * 100))
        
        print(f"\n📈 DATA QUALITY SCORE: {quality_score:.1f}/100")
        
        # Issues by type
        issue_types = {}
        for issue in self.report['issues']:
            issue_type = issue['type']
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        print(f"\n🔍 ISSUES BY TYPE:")
        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {issue_type}: {count}")
        
        # Data patterns discovered
        if self.report['data_patterns']:
            print(f"\n🎯 DATA PATTERNS DISCOVERED:")
            for pattern, value in self.report['data_patterns'].items():
                print(f"   • {pattern}: {value}")
        
        # Top recommendations
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(self.report['recommendations'][:15], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
        print("✅ Comprehensive analysis complete!")
        print("="*60)


# Simple wrapper function
def check_data_quality_enhanced(filepath):
    """Enhanced data quality check with 15+ different checks"""
    df = pd.read_csv(filepath, low_memory=False)
    checker = EnhancedDataQualityChecker(df)
    report = checker.check_all()
    return report


