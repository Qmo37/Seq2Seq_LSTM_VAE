# OULAD Dataset

Place the following CSV files from the Open University Learning Analytics Dataset in this directory:

## Required Files

- `studentInfo.csv` - Student demographic information
- `studentVle.csv` - Virtual Learning Environment interaction logs
- `studentAssessment.csv` - Assessment submissions and scores

## Download Instructions

1. Visit: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
2. Click "Download" button
3. Extract the downloaded ZIP file
4. Copy the above three CSV files to this directory

## File Structure

After downloading, this directory should contain:
```
data/raw/
├── README.md (this file)
├── studentInfo.csv
├── studentVle.csv
└── studentAssessment.csv
```

## Data Description

**studentInfo.csv**
- Student demographic and registration information
- Columns: id_student, code_module, code_presentation, gender, region, etc.

**studentVle.csv**
- Virtual Learning Environment click logs
- Columns: id_student, id_site, date, sum_click
- Records daily student interactions with course materials

**studentAssessment.csv**
- Assessment submission records and scores
- Columns: id_student, id_assessment, date_submitted, score
- Contains homework/exam submissions and grades

## Data Size

Expected file sizes (approximate):
- studentInfo.csv: ~5-10 MB
- studentVle.csv: ~50-100 MB (largest file)
- studentAssessment.csv: ~10-20 MB

## License

The Open University Learning Analytics Dataset is provided for educational and research purposes. Please cite the original source when using this data.
