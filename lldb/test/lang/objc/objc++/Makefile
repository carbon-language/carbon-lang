LEVEL = ../../../make

OBJCXX_SOURCES := main.mm

include $(LEVEL)/Makefile.rules

LDFLAGS += -framework Foundation

main.o:	main.mm
	$(CXX) $(CXXFLAGS) -c -o main.o main.mm
