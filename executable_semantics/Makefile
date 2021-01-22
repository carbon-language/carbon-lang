# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

a.out: syntax.l syntax.ypp ast.h typecheck.h interp.h ast.cpp typecheck.cpp interp.cpp cons_list.h assoc_list.h
	bison -d syntax.ypp
	flex syntax.l
	bison -v --debug syntax.ypp  -o syntax.tab.cpp
	g++ -std=c++11 -Wno-deprecated-register -c -g lex.yy.c
	g++ -std=c++11 -c -g ast.cpp
	g++ -std=c++11 -c -g typecheck.cpp
	g++ -std=c++11 -c -g interp.cpp
	g++ -std=c++11 -g lex.yy.o ast.o typecheck.o interp.o syntax.tab.cpp

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
	rm -f syntax.tab.cpp syntax.tab.c syntax.output
	rm -f lex.yy.c lex.yy.cpp
	rm -f a.out
	rm -rf a.out.dSYM
	rm -f log
	rm -f *~
	rm -f examples/*~
