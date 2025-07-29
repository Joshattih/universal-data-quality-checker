# Universal Data Quality Checker

A comprehensive Python tool that automatically checks data quality issues in ANY dataset. Perfect for data scientists, analysts, and anyone working with data.

## 🚀 Features

✅ **15+ Quality Checks:**
- Missing values analysis
- Duplicate detection
- Data type validation
- Outlier detection
- Formatting inconsistencies
- Invalid values (negative ages, future dates)
- High correlations
- Data freshness
- String pattern detection (emails, phones, URLs)
- Special character detection
- Data consistency checks
- PII detection
- And more!

✅ **Visual Reports:**
- Missing value heatmaps
- Distribution plots
- Correlation matrices
- Issue severity breakdowns

✅ **Smart Recommendations:**
- Context-aware suggestions
- Severity-based prioritization
- Actionable insights

## 📦 Import and use 


# Import and use
import sys
sys.path.append('/content/')  # Add current directory to path

from data_quality_checker import check_data_quality_enhanced

# Check your data
report = check_data_quality_enhanced('Your_data')
