CLANG_FORMAT=/proj/pgi/flang/x86_64/flang-dev/bin/clang-format

Debug Release:
	@mkdir -p $@
	cd $@ && cmake -DCMAKE_BUILD_TYPE=$@ .. && make
.PHONY: Debug Release

reformat:
	@find . -regextype posix-extended -regex '.*\.(h|cc)' \
		| while read file; do \
			echo $$file; \
			$(CLANG_FORMAT) -i $$file; \
		done
.PHONY: reformat
