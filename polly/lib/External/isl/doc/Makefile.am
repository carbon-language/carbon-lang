if GENERATE_DOC
export TEXINPUTS := $(srcdir):$(TEXINPUTS)
export BIBINPUTS := $(srcdir):$(BIBINPUTS)
export BSTINPUTS := $(srcdir):$(BSTINPUTS)

user.tex: user.pod
	$(PERL) $(srcdir)/mypod2latex $< $@
manual.pdf: manual.tex user.tex $(srcdir)/implementation.tex
	(cd ..; echo "@GIT_HEAD_VERSION@") > version.tex
	$(PDFLATEX) $<
	bibtex manual
	$(PDFLATEX) $<
	$(PDFLATEX) $<
user.html: user.pod
	(cd ..; echo "@GIT_HEAD_VERSION@") > version
	$(POD2HTML) --infile=$< --outfile=$@ --title="Integer Set Library: Manual [version `cat version`]"
endif
