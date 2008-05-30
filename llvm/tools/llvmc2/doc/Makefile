
RST2HTML=rst2html

all : LLVMC-Reference.html LLVMC-Tutorial.html

LLVMC-Tutorial.html : LLVMC-Tutorial.rst
	$(RST2HTML) $< $@

LLVMC-Reference.html : LLVMC-Reference.rst
	$(RST2HTML) $< $@

clean :
	rm *.html
