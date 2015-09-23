LEVEL = ../../../make

CXX_SOURCES = main.cpp length.cpp

CFLAGS_LIMIT = -g -O0 -c
CFLAGS_NO_LIMIT = -g -O0 -c

ifneq (,$(findstring clang,$(CC)))
  CFLAGS_LIMIT += -flimit-debug-info
  CFLAGS_NO_LIMIT += -fno-limit-debug-info
endif

all: limit nolimit

limit: main.o length_limit.o
	$(CXX) main.o length_limit.o -o limit

nolimit: main.o length_nolimit.o
	$(CXX) main.o length_nolimit.o -o nolimit

main.o: main.cpp
	$(CXX) $(CFLAGS_LIMIT) main.cpp -o main.o

length_limit.o: length.cpp
	$(CXX) $(CFLAGS_LIMIT) length.cpp -o length_limit.o

length_nolimit.o: length.cpp
	$(CXX) $(CFLAGS_NO_LIMIT) length.cpp -o length_nolimit.o

clean: OBJECTS += limit nolimit length_limit.o length_nolimit.o

include $(LEVEL)/Makefile.rules
