
RST2HTML=rst2html --stylesheet=llvm.css --link-stylesheet

all : LLVMC-Reference.html LLVMC-Tutorial.html

LLVMC-Tutorial.html : LLVMC-Tutorial.rst llvm.css
	$(RST2HTML) $< $@

LLVMC-Reference.html : LLVMC-Reference.rst llvm.css
	$(RST2HTML) $< $@

clean :
	rm *.html
