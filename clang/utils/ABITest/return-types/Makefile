# Usage: make test.N.report 
#
# COUNT can be over-ridden to change the number of tests generated per
# file, and TESTARGS is used to change the type generation. Make sure
# to 'make clean' after changing either of these parameters.

ABITESTGEN := ../ABITestGen.py
TESTARGS := --max-args 0 --no-vector --no-complex --no-union
COUNT := 100

CFLAGS := -std=gnu99

X_COMPILER := llvm-gcc
X_LL_CFLAGS := -emit-llvm -S
Y_COMPILER := clang
Y_LL_CFLAGS := -emit-llvm
CC := gcc

X_CFLAGS := -m32
Y_CFLAGS := -arch i386
CC_CFLAGS := -m32

.PHONY: test.%.report
test.%.report: test.%.xx.diff test.%.xy.diff test.%.yx.diff test.%.yy.diff
	@for t in $^; do \
		if [ -s $$t ]; then \
			echo "TEST $*: $$t failed"; \
		fi; \
	done

.PHONY: test.%.build
test.%.build: test.%.ref test.%.xx test.%.xy test.%.yx test.%.yy test.%.x.defs test.%.y.defs
	@true

###

.PRECIOUS: test.%.xx.diff
test.%.xx.diff: test.%.ref.out test.%.xx.out
	-diff $^ > $@
.PRECIOUS: test.%.xy.diff
test.%.xy.diff: test.%.ref.out test.%.xy.out
	-diff $^ > $@
.PRECIOUS: test.%.yx.diff
test.%.yx.diff: test.%.ref.out test.%.yx.out
	-diff $^ > $@
.PRECIOUS: test.%.yy.diff
test.%.yy.diff: test.%.ref.out test.%.yy.out
	-diff $^ > $@
.PRECIOUS: test.%.defs.diff
test.%.defs.diff: test.%.x.defs test.%.y.defs
	zipdifflines \
	  --replace "%struct.T[0-9]+" "%struct.s" \
	  --replace "byval align [0-9]+" "byval" \
	  $^ > $@

.PRECIOUS: test.%.out
test.%.out: test.%
	-./$< > $@

.PRECIOUS: test.%.ref
test.%.ref: test.%.driver.ref.o test.%.a.ref.o test.%.b.ref.o
	$(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.xx
test.%.xx: test.%.driver.ref.o test.%.a.x.o test.%.b.x.o
	$(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.xy
test.%.xy: test.%.driver.ref.o test.%.a.x.o test.%.b.y.o
	$(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.yx
test.%.yx: test.%.driver.ref.o test.%.a.y.o test.%.b.x.o
	$(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^
.PRECIOUS: test.%.yy
test.%.yy: test.%.driver.ref.o test.%.a.y.o test.%.b.y.o
	$(CC) $(CFLAGS) $(CC_CFLAGS) -o $@ $^

.PRECIOUS: test.%.ref.o
test.%.ref.o: test.%.c
	$(CC) -c $(CFLAGS) $(CC_CFLAGS) -o $@ $<
.PRECIOUS: test.%.x.o
test.%.x.o: test.%.c
	$(X_COMPILER) -c $(CFLAGS) $(X_CFLAGS) -o $@ $<
.PRECIOUS: test.%.y.o
test.%.y.o: test.%.c
	$(Y_COMPILER) -S $(CFLAGS) $(Y_CFLAGS) -o $@.s $<
	as $(Y_CFLAGS) -o $@ $@.s

.PRECIOUS: test.%.x.defs
test.%.x.defs: test.%.a.x.ll
	grep '^define ' $< > $@
.PRECIOUS: test.%.y.defs
test.%.y.defs: test.%.a.y.ll
	grep '^define ' $< > $@

.PRECIOUS: test.%.a.x.ll
test.%.a.x.ll: test.%.a.c
	$(X_COMPILER) $(CFLAGS) $(X_LL_CFLAGS) $(X_CFLAGS) -o $@ $<
.PRECIOUS: test.%.b.x.ll
test.%.b.x.ll: test.%.b.c
	$(X_COMPILER) $(CFLAGS) $(X_LL_CFLAGS) $(X_CFLAGS) -o $@ $<
.PRECIOUS: test.%.a.y.ll
test.%.a.y.ll: test.%.a.c
	$(Y_COMPILER) $(CFLAGS) $(Y_LL_CFLAGS) $(Y_CFLAGS) -o $@ $<
.PRECIOUS: test.%.b.y.ll
test.%.b.y.ll: test.%.b.c
	$(Y_COMPILER) $(CFLAGS) $(Y_LL_CFLAGS) $(Y_CFLAGS) -o $@ $<

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
