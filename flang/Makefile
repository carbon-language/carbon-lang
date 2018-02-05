CXX=g++
CXXFLAGS=-Wall -std=c++17 -O2
# CXXFLAGS=-Wall -std=c++17 -g

# CXX=/proj/pgi/linux86-64/dev/bin/pgc++
# CXXFLAGS=-std=c++17

SRCS=f2018-demo.cc char-buffer.cc idioms.cc message.cc \
     parse-tree.cc position.cc preprocessor.cc prescan.cc source.cc
RELS=$(SRCS:.cc=.o)

f18: $(RELS)
	$(CXX) -o $@ $(RELS)
clean:
	rm -f *.o core.* perf.data* out
clobber: clean
	rm -f f18 *~
backup: clean
	(cd ../..; tar czf ~/save/f18.tgz.`date '+%m-%d-%y'` .)
	(cd ../..; tar czf /local/home/save/f18.tgz.`date '+%m-%d-%y'` .)

f2018-demo.o: basic-parsers.h char-buffer.h cooked-chars.h \
              grammar.h idioms.h parse-tree.h prescan.h source.h user-state.h
char-buffer.o: char-buffer.h idioms.h
idioms.o: idioms.h
message.o: message.h
parse-tree.o: parse-tree.h idioms.h indirection.h
preprocessor.o: preprocessor.h idioms.h
prescan.o: prescan.h char-buffer.h idioms.h source.h
position.o: position.h
source.o: source.h char-buffer.h idioms.h

basic-parsers.h: idioms.h message.h parse-state.h position.h
	@touch $@
char-parsers.h: basic-parsers.h parse-state.h
	@touch $@
cooked-chars.h: basic-parsers.h char-parsers.h idioms.h parse-state.h
	@touch $@
cooked-tokens.h: basic-parsers.h cooked-chars.h idioms.h position.h
	@touch $@
debug-parser.h: basic-parsers.h parse-state.h
	@touch $@
prescan.h: char-buffer.h preprocessor.h source.h
	@touch $@
grammar.h: basic-parsers.h cooked-chars.h cooked-tokens.h \
           format-specification.h parse-tree.h user-state.h
	@touch $@
message.h: position.h
	@touch $@
parse-state.h: message.h position.h
	@touch $@
parse-tree.h: format-specification.h idioms.h indirection.h position.h
	@touch $@


CLANG_FORMAT=/proj/pgi/flang/x86_64/flang-dev/bin/clang-format
formatted:
	@mkdir -p formatted
	@for x in *.h *.cc; do \
		$(CLANG_FORMAT) < $$x > formatted/$$x; \
	done

.PHONY: formatted
