# General Makefile containing all the necessary compiler flags for the tests

# modify CC and CFLAGS for OpenMP C compilers
# modify FC and FFLAGS for OpenMP Fortran compilers


# Content:
#########################################################

# 1. Basic usage
# 2. General testsuite settings
# 3. Compiler selection and Flags

#########################################################


#########################################################
# 1. Basic usage
#########################################################
# 	make ctest		generate c test "ctest"
# 	make ftest		generate fortran test "ftest"
#	make clean		removes all sources and binaries
# 	make cleanall	removes sources,binaries and logfiles


#########################################################
# 2. General testsuite settings
#########################################################

# For general testsuite settings see the configuration file
# ompts.conf

#########################################################
# 3. Compiler selection and Flags
#########################################################

# CLANG Compiler
CC     = clang
CFLAGS = -fopenmp -S -emit-llvm
FC     = gfortran
FFLAGS = -fopenmp -lm
FFLAGS = -fopenmp -lm -O3

# GNU Compiler
# CC     = gcc
# CFLAGS = -fopenmp -lm -O3
# FC     = gfortran 
# FFLAGS = -fopenmp -lm
# FFLAGS = -fopenmp -lm -O3

# Fujitsu Compilers:
#CC = fcc
#CFLAGS = -KOMP,fast_GP2=2
#FC=frt
#FFLAGS=-KOMP,fast_GP2=2 -w -Am -X9 -Fixed


# PGI compilers
#CC = pgcc
#CFLAGS = -mp
#CFLAGS = -mp -DVERBOSE
#CFLAGS = -fast -mp

#FC = pgf90
#FFLAGS = -fast -mp
#FFLAGS = -mp -g


# Intel compilers:
#CC = icc
#CFLAGS = -O3 -ip -openmp
#CFLAGS = -Wall -O0 -openmp
#CFLAGS =  -openmp -lm
#CFLAGS =  -openmp -lm -DVERBOSE

#FC = ifort
#FFLAGS = -openmp -lm -fpp

# Omni compilers:
#CC = ompcc
#CFLAGS = -O3 -lm


# Assure compilers:
#CC = assurec
#CFLAGS = -O3 -WApname=project -DUSE_ASSURE=1
#FC =
#FFLAGS =

# NEC:
#CC = c++
#CC = sxc++
#CFLAGS = -Popenmp

#FC=sxf90
#FFLAGS= -Popenmp


# Hitachi:
#CC = xcc
#CFLAGS = -O4 -pvec +Op -parallel -omp
#FC =
#FFLAGS =


# SGI:
#CC = cc
#CFLAGS = -mp -lm
#FC =
#FFLAGS =


# IBM compilers:
#CC = xlc_r
#CFLAGS = -qsmp=omp -lm


#FC=xlf90_r
#FFLAGS=-qsmp=omp -qfixed=132 -qlanglvl=extended


# SUN compilers
#CC = cc
#CFLAGS = -fast -xopenmp -lm

#FC =f90
#FFLAGS = -xopenmp -fast -lm


# open64 compilers
# remark: -I. was a workaround because the installation came without omp.h file
#CC = opencc
#CFLAGS = -O0 -openmp -lm -I. -lomp -lpthread
#CFLAGS = -O0 -openmp -lm -I /home/matthew/opt/usr/include -lpthread
#CFLAGS = -openmp -lm

#FC = openf90
#FFLAGS = -O0 -openmp -lm  -lomp -lpthread


#Pathscale Compiler
#CC = pathcc
#CFLAGS = -mp -Ofast -lm

#FC=pathf90
#FFLAGS= -mp -Ofast -lm


#OpenUH Compiler
#CC = uhcc
#CFLAGS = -mp

#FC = uhf90
#FFLAGS = -mp

# Check path
FILE=LLVM-IR/lit.tmp
CHECK_PATH=`cat $(FILE)`

#########################################################################

help:
	@echo "OpenMP Validation Suite, v3.0"
	@echo "(C) University of Stuttgart, University of Houston"
	@echo ""
	@echo "Do make [rule], where rule may be one of:"
	@echo "  ctest"
	@echo "    Build and run the OpenMP/C validation tests."
	@echo "  ftest"
	@echo "    Build and run the OpenMP/Fortran validation tests."
	@echo "  print_compile_options"
	@echo "    Display the default compiler and compiler options being tested"
	@echo "  cleansrcs"
	@echo "    Remove sources generated from the templates"
	@echo "  cleanbins"
	@echo "    Remove executables from bin/ directory"
	@echo "  cleanouts"
	@echo "    Remove all *.out files from bin/ directory"
	@echo "  cleanlogs"
	@echo "    Remove all *.log files from bin/ directory"
	@echo "  clean"
	@echo "    Clean out and log files"
	@echo "  distclean"
	@echo "    Clean arch test directory with clean"
	@echo "  veryclean"
	@echo "    Remove the entire bin/c directory with distclean"

omp_my_sleep:
	mkdir -p bin/c
	cp omp_my_sleep.h bin/c/
omp_testsuite: omp_testsuite.h
	mkdir -p bin/c
	cp omp_testsuite.h bin/c/
omp_testsuite.h: ompts-c.conf c/*
	./ompts_makeHeader.pl -f=ompts-c.conf -t=c
.c.o: omp_testsuite omp_my_sleep
	$(CC) $(CFLAGS) -c $<

ctest: omp_my_sleep omp_testsuite
	./runtest.pl --lang=c testlist-c.txt
	cd bin/; make -j

ftest:
	mkdir -p bin/fortran
	cp omp_testsuite.f bin/fortran
	cp omp_my_sleep.f bin/fortran
	./runtest.pl --lang=fortran testlist-f.txt

print_compile_options:
	@echo "-------------------"
	@echo "C compiler"
	@echo "-------------------"
	@echo "compilation: $(CC) $(CFLAGS)"
	@$(CC) --version
	@echo "-------------------"
	@echo "Fortran compiler"
	@echo "-------------------"
	@echo "compilation: $(FC) $(FFLAGS)"
	@$(FC) --version



path_to_arch=$(shell ../runtime/tools/check-openmp-test.pl)

cleansrcs:
	find bin/ -iname "*.[cf]" -exec rm -rf {} \;
cleanbins:
	find bin/c -perm /a+x -type f -exec rm -rf {} \;
	find ./bin/c -not -name "*.c" -a -not -name "*.ll" -type f -exec rm -rf {} \;
cleanouts:
	find bin/ -iname "*.out" -exec rm -rf {} \;
cleanlogs:
	find bin/ -iname "*.log" -exec rm -rf {} \;
	rm -rf ./$(path_to_arch)

clean: cleanouts cleanlogs cleanbins cleansrcs

distclean: clean
	rm -rf bin/c/*
	rm -rf LLVM-IR/$(CHECK_PATH)

veryclean: distclean
	rm -rf bin/c/
	rm -rf bin/fortran/


cleanall: 
	rm -rf ./bin/c/
	rm -rf ./bin/fortran/
	rm -rf ./LLVM-IR/$(path_to_arch)
	rm -rf ./$(path_to_arch)
	rm -rf ./*.log


