# Usage: make test.N.report 
#
# COUNT can be over-ridden to change the number of tests generated per
# file, and TESTARGS is used to change the type generation. Make sure
# to 'make clean' after changing either of these parameters.

ABITESTGEN := ../ABITestGen.py
TESTARGS := --max-args 0
COUNT := 1
TIMEOUT := 5

CFLAGS := -std=gnu99

X_COMPILER := llvm-gcc
X_LL_CFLAGS := -emit-llvm -S
Y_COMPILER := xcc -ccc-clang
Y_LL_CFLAGS := -emit-llvm -S
CC := gcc

ifeq (0, 1)
X_CFLAGS := -m32
Y_CFLAGS := -m32
CC_CFLAGS := -m32
else
X_CFLAGS := -m64
Y_CFLAGS := -m64
CC_CFLAGS := -m64
endif

ifndef VERBOSE
  Verb := @
endif

.PHONY: test.%.report
test.%.report: test.%.xx.diff test.%.xy.diff test.%.yx.diff test.%.yy.diff
	@ok=1;\
	for t in $^; do \
		if [ -s $$t ]; then \
			echo "TEST $*: $$t failed"; \
			ok=0;\
		fi; \
	done; \
	if [ $$ok == 1 ]; then \
		echo "TEST $*: OK"; \
	else \
		false; \
	fi


.PHONY: test.%.defs-report
test.%.defs-report: test.%.defs.diff
	@for t in $^; do \
		if [ -s $$t ]; then \
			echo "TEST $*: $$t failed"; \
			cat $$t; \
		fi; \
	done

.PHONY: test.%.build
test.%.build: test.%.ref test.%.xx test.%.xy test.%.yx test.%.yy test.%.x.defs test.%.y.defs
	@true

###

.PRECIOUS: test.%.xx.diff
test.%.xx.diff: test.%.ref.out test.%.xx.out
	$(Verb) diff $^ > $@ || true
.PRECIOUS: test.%.xy.diff
test.%.xy.diff: test.%.ref.out test.%.xy.out
	$(Verb) diff $^ > $@ || true
.PRECIOUS: test.%.yx.diff
test.%.yx.diff: test.%.ref.out test.%.yx.out
	$(Verb) diff $^ > $@ || true
.PRECIOUS: test.%.yy.diff
test.%.yy.diff: test.%.ref.out test.%.yy.out
	$(Verb) diff $^ > $@ || true
.PRECIOUS: test.%.defs.diff
test.%.defs.diff: test.%.x.defs test.%.y.defs
	$(Verb) zipdifflines \
	  --replace "%struct.T[0-9]+" "%struct.s" \
	  --replace "byval align [0-9]+" "byval" \
	  $^ > $@

.PRECIOUS: test.%.out
test.%.out: test.%
	$(Verb) -RunSafely.sh $(TIMEOUT) 1 /dev/null $@ ./$<

.PRECIOUS: test.%.ref
test.%.ref: test.%.driver.ref.o test.%.a.ref.o test.%.b.ref.o
	$(Verb) $(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.xx
test.%.xx: test.%.driver.ref.o test.%.a.x.o test.%.b.x.o
	$(Verb) $(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.xy
test.%.xy: test.%.driver.ref.o test.%.a.x.o test.%.b.y.o
	$(Verb) $(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.yx
test.%.yx: test.%.driver.ref.o test.%.a.y.o test.%.b.x.o
	$(Verb) $(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.yy
test.%.yy: test.%.driver.ref.o test.%.a.y.o test.%.b.y.o
	$(Verb) $(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^

.PRECIOUS: test.%.ref.o
test.%.ref.o: test.%.c
	$(Verb) $(CC) -c $(CFLAGS) $(CC_CFLAGS) -o $@ $<
.PRECIOUS: test.%.x.o
test.%.x.o: test.%.c
	$(Verb) $(X_COMPILER) -c $(CFLAGS) $(X_CFLAGS) -o $@ $<
.PRECIOUS: test.%.y.o
test.%.y.o: test.%.c
	$(Verb) $(Y_COMPILER) -c $(CFLAGS) $(Y_CFLAGS) -o $@ $<

.PRECIOUS: test.%.x.defs
test.%.x.defs: test.%.a.x.ll
	-$(Verb) -grep '^define ' $< > $@
.PRECIOUS: test.%.y.defs
test.%.y.defs: test.%.a.y.ll
	-$(Verb) -grep '^define ' $< > $@

.PRECIOUS: test.%.a.x.ll
test.%.a.x.ll: test.%.a.c
	$(Verb) $(X_COMPILER) $(CFLAGS) $(X_LL_CFLAGS) $(X_CFLAGS) -o $@ $<
.PRECIOUS: test.%.b.x.ll
test.%.b.x.ll: test.%.b.c
	$(Verb) $(X_COMPILER) $(CFLAGS) $(X_LL_CFLAGS) $(X_CFLAGS) -o $@ $<
.PRECIOUS: test.%.a.y.ll
test.%.a.y.ll: test.%.a.c
	$(Verb) $(Y_COMPILER) $(CFLAGS) $(Y_LL_CFLAGS) $(Y_CFLAGS) -o $@ $<
.PRECIOUS: test.%.b.y.ll
test.%.b.y.ll: test.%.b.c
	$(Verb) $(Y_COMPILER) $(CFLAGS) $(Y_LL_CFLAGS) $(Y_CFLAGS) -o $@ $<

.PHONY: test.%.top
test.%.top: test.%.a.c test.%.b.c test.%.driver.c
	@true

.PRECIOUS: test.%.a.c test.%.b.c test.%.driver.c
test.%.a.c: test.%.generate
	@true
test.%.b.c: test.%.generate
	@true
test.%.driver.c: test.%.generate
	@true

.PHONY: test.%.generate
test.%.generate: $(ABITESTGEN)
	$(ABITESTGEN) $(TESTARGS) -o test.$*.a.c -T test.$*.b.c -D test.$*.driver.c --min=$(shell expr $* '*' $(COUNT))  --count=$(COUNT)

clean:	
	rm -f test.* *~
