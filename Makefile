#

# usage: make all

# run all analysis

all: documents/project_report.md

# running the model
models: src/analysis4make.py
	python src/analysis4make.py

# create plots and csv files to be transferred to R
figures_files: src/exploratory.py models
	python src/exploratory.py

# make project_report
documents/project_report.md: src/project_report.Rmd figures_files
	Rscript -e "ezknitr::ezknit('src/project_report.Rmd', out_dir = 'documents')"

#Clean up intermediate files

clean:

	rm -f data/stat_smry_consp.csv
	rm -f data/stat_smry_traits.csv
	rm -f doc/count_report.md doc/count_report.html
