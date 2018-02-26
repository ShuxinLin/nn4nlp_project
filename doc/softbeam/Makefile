TEX = softbeam
BIB = MachineLearning
all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex $(BIB).bib
	pdflatex --shell-escape $(TEX).tex
	bibtex $(TEX)
	pdflatex --shell-escape $(TEX).tex
	pdflatex --shell-escape $(TEX).tex

CLEAN_FILES = *.aux *.log *.out *.synctex.gz *.bbl *.blg $(TEX).pdf
clean:
	rm -f $(CLEAN_FILES)
