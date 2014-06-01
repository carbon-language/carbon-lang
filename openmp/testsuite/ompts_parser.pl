#!/usr/bin/env perl

# ompts_parser [option] SOURCEFILE
# 
# Creats the tests and the crosstests for the OpenMP-Testsuite out of an templatefiles which are given to the programm.
# 
# Options:
# -test: 	make test
# -crosstest: 	make crosstest
# -orphan	if possible generate tests using orphan regions (not implemented yet)
# -lang=LANG	preprocessing for language LANG, where LANG is one of the following languages:
#		c, fortran
# -o=FILENAME	outputfile (only when one templatefile is specified)


# Using Getopt::long to extract the programm options
use Getopt::Long;
# Using functions: Set of subroutines to modify the testcode
use ompts_parserFunctions;

# Getting given options
GetOptions("-test" => \$test,"-crosstest" => \$crosstest, "-o=s" => \$outputfile, "-orphan" => \$orphan, "-f!", "-lang=s" => \$language);

# Remaining arguments are the templatefiles. 
# Adding these to the list of to be parsed files if they exist.
foreach $file(@ARGV)
{
	if(-e $file){ push(@sourcefiles,$file); }
	else { print "Error: Unknown Option $file\n"; }
}
	
# Checking if options were valid:
#################################################################
# preparations and checks for sourcefiles
if(@sourcefiles == 0){die "No files to parse are specified!";}
if($outputfile && (@sourcefiles != 1 || ($test && $crosstest) ) ){die "There were multiple files for one outputfiles specified!";} 
# preparations fopr orphan tests
if($orphan){ $orphanprefix = "orphaned"; }
else { $orphanprefix = ""; }
# preparations for test / crosstest
if($test){push(@testtypes,"test"); 
# %checks['test']="check";
}
if($crosstest){push(@testtypes,"ctest");
# %checks['crosstest']="crosscheck";
}
# preparations and checks for language
if($language eq"c") { $extension = "c";}
elsif($language eq "fortran" or $language eq "f") { $language = "f"; $extension = "f";}
else { die "You must specify a valid language!"; }
    

# Reading the templates for the tests into @sources
foreach $srcfile (@sourcefiles)
{
	# Reading the content of the current sourcefile	into $src
	open(TEST,$srcfile) or print "Error: Could not open template $srcfile\n";
	while(<TEST>){ $src .= $_; }
	close(TEST);
	# Adding the content $src to the end of the list @sources
	push(@sources,$src);
}

# Extracting the source for the mainprogramm and saving it in $mainprocsrc
if($language eq "c") { $mainprocsrc = "ompts_standaloneProc.c"; }
elsif($language eq "f") { $mainprocsrc = "ompts_standaloneProc.f"; } 
open(MAINPROC,$mainprocsrc) or die "Could not open the sourcefile for the main program $mainprocsrc";
while(<MAINPROC>){
	$mainproc .= $_;
}

foreach $testtype (@testtypes)
{
  foreach $src(@sources)
  {
# Some temporary testinformation:
    ($description) = get_tag_values('ompts:testdescription',$src);
    ($directive) = get_tag_values('ompts:directive',$src);
    ($functionname) = get_tag_values('ompts:testcode:functionname',$src);

    open(OUTFILE,">".$language.$orphanprefix.$testtype."_".$functionname.".".$extension) or die("Could not create the output file for $directive");

# Creating the source for the test:
    ($code) = get_tag_values('ompts:testcode',$src);
# Putting together the functions and the mainprogramm:
    $code .= $mainproc;
    
# get the parameters <ompts:orphan:params> by joon
# thanks to Dr. Yin Ma in Absoft
    ($parms) = get_tag_values('ompts:orphan:parms',$code);
    ($parms) = leave_single_space($parms);
# to remove parameters tag between 'ompts:orphan:parms' by joon
    ($code) = replace_tags('ompts:orphan:parms','',$code);
    
# Make modifications for the orphaned testversion if necessary:
    if($orphan)
    {
# Get the global variables:
      @defs = get_tag_values("ompts:orphan:vars",$code);
      $orphvarsdef = "";
      foreach $_ (@defs)
      {
	#print $_;
	if(not /^[ ]*$/gs) { $orphvarsdef = join("\n",$orphvarsdef,$_); } 
	#print "OK\n".$orphvarsdef; 
      }
      if($language eq "f")
      {
# Generate the orphan subroutines:
	$orphfuncs = create_orph_fortranfunctions("$testtype_", $code);
# Replace orphan regions by functioncalls:
	($code) = orphan_regions2fortranfunctions( "$testtype_", ($code) );
	($code) = enlarge_tags('ompts:orphan:vars','','',($code));
    ($code) = enlarge_tags('ompts:orphan:parms','','',($code));
    #to find orphan call statemetn and add parameters
    
# Put all together:
	$code = $code . $orphfuncs;
      }
      elsif($language eq "c")
      {
# Generate predeclarations for orpahn functions:
	$orphfuncsdefs = orph_functions_declarations("$testtype_",$code);
# Generate the orphan functions:
	$orphfuncs = create_orph_cfunctions("$testtype_",$code);
# Repla:e orphan regions by functioncalls:
	($code) = orphan_regions2cfunctions( "$testtype_", ($code) );
# Deleting the former declarations of the variables in the orphan regions:
	($code) = delete_tags('ompts:orphan:vars',($code));
# Put all together:
	$code = "#include \"omp_testsuite.h\"\n".$orphvarsdef . $orphfuncsdefs . $code . $orphfuncs;
      }
      else {
	print "An error occurred!";
      }
    }
# remove parameters between <ompts:orphan:parms> tags, added by joon
    ($code)= replace_tags('ompts:orphan:parms',$code);
    
# Remove the marks for the orpahn regions and its variables:
    ($code) = enlarge_tags('ompts:orphan','','',($code));
    ($code) = enlarge_tags('ompts:orphan:vars','','',($code));

# remove parameters between for orphaned directive parametes, added by joon
    ($code) = enlarge_tags('ompts:orphan:parms','','',($code));
    
    if($testtype eq "test") {
# Remove the marks for the testcode and remove the code for the crosstests: 
      ($code) = enlarge_tags('ompts:check','','',($code));
      ($code) = delete_tags('ompts:crosscheck',($code));		
    }
    elsif($testtype eq "ctest") {
# Remove the marks for the crosstestcode and remove the code for the tests: 
      ($code) = enlarge_tags('ompts:crosscheck','','',($code));
      ($code) = delete_tags('ompts:check',($code));		
    }
# Making some final modifications:
    ($code) = replace_tags('testfunctionname',$testtype."_".$functionname,($code));
    ($code) = replace_tags('directive',$directive,($code));
    ($code) = replace_tags('description',$description,($code));
    ($code) = enlarge_tags('ompts:testcode:functionname',$testtype."_",'',($code) );
#	$code =  "\#include \"omp_testsuite.h\"\n".$code;
# Write the result into the file and close it:
    print OUTFILE $code;
    close(OUTFILE);
  }
}
