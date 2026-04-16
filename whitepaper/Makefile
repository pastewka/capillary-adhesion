QUARTO = quarto
SRC    = whitepaper.qmd

.PHONY: all html pdf clean

all: html pdf

html: $(SRC)
	$(QUARTO) render $(SRC) --to html

pdf: $(SRC)
	$(QUARTO) render $(SRC) --to pdf

clean:
	rm -f whitepaper.html whitepaper.pdf whitepaper.tex
	rm -rf whitepaper_files
