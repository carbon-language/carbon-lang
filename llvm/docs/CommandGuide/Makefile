POD = $(wildcard *.pod)
HTML = $(patsubst %.pod, html/%.html, $(POD))
MAN = $(patsubst %.pod, man/man1/%.1, $(POD))
PS = $(patsubst %.pod, ps/%.ps, $(POD))

all: $(HTML) $(MAN) $(PS)

.SUFFIXES:
.SUFFIXES: .html .pod .1 .ps

html/%.html: %.pod
	pod2html --css=manpage.css --htmlroot=. \
		--podpath=. --noindex --infile=$< --outfile=$@

man/man1/%.1: %.pod
	pod2man --release=1.3 --center="LLVM User's Guide" $< $@

ps/%.ps: man/man1/%.1
	groff -Tps -man $< > $@

clean:
	rm -f pod2htm*.*~~ $(HTML) $(MAN) $(PS)

