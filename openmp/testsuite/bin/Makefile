CC     = clang
CFLAGS = -fopenmp -S -emit-llvm
INCL =
LIBS =

BIN =
SRC_DIR = ./c
SRC_FILES := $(foreach DIR, $(SRC_DIR), $(wildcard $(SRC_DIR)/*.c))
OBJS := $(patsubst %.c, %.o, $(SRC_FILES))
TARGETLIST := $(patsubst %.c, %, $(SRC_FILES))

all:$(TARGETLIST)
	./distribute.sh
	@ echo all done

.PHONY: clean

$(TARGETLIST): $(SRC_FILES)
	$(CC) $(CFLAGS) $(LIBS) $(INCL) $@.c -o $@.ll

clean:
	@ rm -rf c/*
	@ echo clean bin done
