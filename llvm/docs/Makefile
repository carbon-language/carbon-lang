doxygen-files:
	doxygen doxygen.cfg

doxygen.tar.gz: doxygen/index.html
	rm -rf $@
	gtar czvf $@ doxygen
