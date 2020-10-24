a.out: syntax.l syntax.y ast.h typecheck.h interp.h ast.cc typecheck.cc interp.cc cons_list.h assoc_list.h 
	bison -d syntax.y
	flex syntax.l
	bison -v --debug syntax.y  -o syntax.tab.cc
	g++ -std=c++11 -Wno-deprecated-register -c -g lex.yy.c
	g++ -std=c++11 -c -g ast.cc
	g++ -std=c++11 -c -g typecheck.cc
	g++ -std=c++11 -c -g interp.cc
	g++ -std=c++11 -g lex.yy.o ast.o typecheck.o interp.o syntax.tab.cc

# TODO: replace this with a script that can handle
#  tests that are suppose to produce errors.
test:
	./a.out examples/zero.6c
	./a.out examples/next.6c
	./a.out examples/while1.6c
	./a.out examples/fun1.6c
	./a.out examples/fun2.6c
	./a.out examples/fun3.6c
	./a.out examples/fun4.6c
	./a.out examples/fun5.6c
	./a.out examples/fun-recur.6c
	./a.out examples/funptr1.6c
	./a.out examples/block1.6c
	./a.out examples/break1.6c
	./a.out examples/continue1.6c
	./a.out examples/struct1.6c
	./a.out examples/struct2.6c
	./a.out examples/choice1.6c
	./a.out examples/tuple1.6c
	./a.out examples/tuple2.6c
	./a.out examples/tuple-match.6c
	./a.out examples/tuple-assign.6c
	./a.out examples/record1.6c
	./a.out examples/match-int.6c
	./a.out examples/match-int-default.6c
	./a.out examples/match-type.6c
	./a.out examples/pattern-init.6c
	./a.out examples/struct3.6c

clean:
	rm -f a.out
	rm -f *.o
	rm -f syntax.tab.h
	rm -f syntax.tab.cc syntax.tab.c syntax.output
	rm -f lex.yy.c lex.yy.cc
	rm -f a.out
	rm -rf a.out.dSYM
	rm -f log
	rm -f *~
	rm -f examples/*~
