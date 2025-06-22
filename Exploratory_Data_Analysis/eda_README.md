# Exploratory Data Analysis (EDA) Project

A comprehensive Python toolkit for performing exploratory data analysis on datasets, featuring automated statistical analysis, visualization, and reporting.

## 📋 Project Overview

This project provides a complete EDA solution that:
- Analyzes dataset structure and basic statistics
- Identifies and visualizes missing values
- Performs univariate and bivariate analysis
- Detects outliers using statistical methods
- Generates correlation analysis and heatmaps
- Creates comprehensive visual reports

## 🚀 Features

- **Automated Analysis**: Complete EDA pipeline with one function call
- **Missing Value Detection**: Comprehensive missing data analysis
- **Statistical Summaries**: Descriptive statistics for numerical variables
- **Visualization Suite**: Histograms, box plots, correlation heatmaps, and more
- **Outlier Detection**: IQR-based outlier identification
- **Report Generation**: Automated summary reports with key findings

## 📁 Project Structure

```
eda-project/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── eda_analyzer.py
├── notebooks/
│   └── eda_demo.ipynb
├── data/
│   └── sample_data.csv
└── reports/
    ├── figures/
    └── eda_report.txt
```

## 🛠️ Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Basic Usage

```python
from src.eda_analyzer import DataAnalyzer

# Initialize with your data
analyzer = DataAnalyzer(data_path='your_data.csv')

# Or with a DataFrame
analyzer = DataAnalyzer(dataframe=your_df)

# Generate complete EDA report
analyzer.generate_report()
```

### Individual Analysis Components

```python
# Basic dataset information
analyzer.basic_info()

# Missing values analysis
analyzer.missing_values_analysis()

# Numerical variables analysis
analyzer.numerical_analysis()

# Categorical variables analysis
analyzer.categorical_analysis()

# Correlation analysis
analyzer.correlation_analysis()

# Outlier detection
analyzer.outlier_detection()
```

### Run with Sample Data

```bash
python src/eda_analyzer.py
```

## 📊 Generated Outputs

The EDA analyzer creates the following visualizations:

1. **Missing Values Plot**: Bar chart showing missing data percentage
2. **Numerical Distributions**: Histograms with KDE for all numerical variables
3. **Categorical Distributions**: Pie charts showing category distributions
4. **Correlation Heatmap**: Correlation matrix visualization
5. **Outlier Box Plots**: Box plots highlighting outliers
6. **Summary Report**: Text file with key findings and recommendations

## 🔧 Key Features

### Statistical Analysis
- Descriptive statistics (mean, median, std, quartiles)
- Distribution analysis with normality tests
- Correlation analysis with significance testing
- Outlier detection using IQR method

### Visualization
- Automated plot generation for all variable types
- Customizable color schemes and styling
- High-resolution output suitable for reports
- Clear labeling and statistical annotations

### Reporting
- Automated summary generation
- Key findings identification
- Data quality assessment
- Actionable recommendations

## 📈 Sample Results

### Dataset Overview
- **Shape**: 1000 rows × 8 columns
- **Missing Values**: 5% in satisfaction_score
- **Variable Types**: 5 numerical, 3 categorical
- **Memory Usage**: ~64 KB

### Key Insights
- Strong correlation between age and experience (r=0.78)
- Income shows log-normal distribution
- 15% outliers detected in satisfaction_score
- Balanced distribution across departments

## 🔍 Technical Details

- **Language**: Python 3.7+
- **Core Libraries**: pandas, numpy, matplotlib, seaborn, scipy
- **Statistical Methods**: IQR outlier detection, Pearson correlation
- **Visualization**: Seaborn for statistical plots, matplotlib for customization
- **Output Formats**: PNG images, text reports, CSV summaries

## 🎯 Use Cases

- **Data Quality Assessment**: Identify data issues before modeling
- **Feature Engineering**: Discover relationships for new features
- **Business Insights**: Extract actionable insights from data
- **Report Generation**: Create professional analysis reports
- **Data Preprocessing**: Inform cleaning and transformation decisions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Future Enhancements

- Interactive visualizations with Plotly
- Advanced statistical tests
- Automated feature engineering suggestions
- Integration with popular ML libraries
- Custom report templates
