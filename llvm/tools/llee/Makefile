LEVEL = ../..
include $(LEVEL)/Makefile.config

SRCS = ExecveHandler.c SysUtils.c

OBJS = $(SRCS:%.c=%.o)
SO   = execve.so

all: $(SO) execve_test
	
%.o: %.c
	gcc -g -I../../include -D_GNU_SOURCE $< -c -o $@

$(SO): $(OBJS)
	gcc -g -shared -ldl -rdynamic $(OBJS) -o $@

execve_test: execve_test.c
	gcc -g $< -o $@

clean:
	rm -f $(OBJS) $(SO)
