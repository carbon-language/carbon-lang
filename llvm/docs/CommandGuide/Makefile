POD = $(wildcard *.pod)
HTML = $(patsubst %.pod, %.html, $(POD))
MAN = $(patsubst %.pod, %.1, $(POD))
PS = $(patsubst %.pod, %.ps, $(POD))

all: $(HTML) $(MAN) $(PS)

.SUFFIXES:
.SUFFIXES: .html .pod .1 .ps

%.html: %.pod
	pod2html --css=http://www.perldoc.com/css/perldoc.css \
		--podpath=/`pwd` --noindex --infile=$< --outfile=$@

%.1: %.pod
	pod2man --release=1.3 --center="LLVM User's Guide" $< $@

%.ps: %.1
	groff -Tps -man $< > $@

clean:
	rm -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

