#!/usr/bin/env perl

# ompts_parser [option] INFILE OUTFILE
# 
# Creats the tests and the crosstests for the OpenMP-Testsuite out of an templatefiles which are given to the programm.
# 
# Options:
# --test: 	make test
# --crosstest: 	make crosstest
# --orphan	if possible generate tests using orphan 
#
# Return:
#       Succes:                 0
#       Template not found      -1
#

# Using Getopt::long to extract the programm options
use Getopt::Long;
# Using functions: Set of subroutines to modify the testcode
use ompts_parserFunctions;

# Getting given options
GetOptions("test" => \$test,"crosstest" => \$crosstest, "orphan!" => \$orphan);

# Remaining arguments are the templatefiles. 
# Adding these to the list of to be parsed files if they exist.

my $templatefile;
my $sourcefile;
my $mainprocsrc = "ompts_standaloneProc.c"; 

$templatefile = $ARGV[0];
$outfile = $ARGV[1];

if (!-e $templatefile) {
    print "Temaplte file not found";
    exit -1;
}

	
# Checking if options were valid:
#################################################################
# preparations and checks for sourcefiles

# Reading the template for the tests 
open(TEST,$templatefile) or die "Error: Could not open template $srcfile\n";
while(<TEST>){ $src .= $_; }
close(TEST);

# Extracting the source for the mainprogramm and saving it in $mainprocsrc
open(MAINPROC,$mainprocsrc) or die "Could not open the sourcefile for the main program $mainprocsrc";
while(<MAINPROC>){ $mainproc .= $_; }
close (MAINPROC);

# Some temporary testinformation:
my ($description)  = get_tag_values ('ompts:testdescription',$src);
my ($directive)    = get_tag_values ('ompts:directive',$src);
my ($functionname) = get_tag_values ('ompts:testcode:functionname',$src);

open (OUTFILE,">$outfile") or die "Could not create the output file for $directive";

# Creating the source for the test:
my ($code) = get_tag_values('ompts:testcode',$src);
# Putting together the functions and the mainprogramm:
$code .= $mainproc;

my $testprefix = "";

# Make modifications for the orphaned testversion if necessary:
if ($orphan) {
# Get the global variables:
    @defs = get_tag_values("ompts:orphan:vars",$code);
    $orphvarsdef = "";
    foreach (@defs) {
        $orphvarsdef = join("\n",$orphvarsdef,$_); 
    }
# Generate predeclarations for orpahn functions:
    $orphfuncsdefs = orph_functions_declarations($code);
# Generate the orphan functions:
    $orphfuncs = create_orph_cfunctions($code);
# Repla:e orphan regions by functioncalls:
    $code = orphan_regions2cfunctions($code);
# Deleting the former declarations of the variables in the orphan regions:
    ($code) = delete_tags('ompts:orphan:vars',($code));
# Put all together:
    $code = "#include \"omp_testsuite.h\"\n" . $orphvarsdef . $orphfuncsdefs . $code . $orphfuncs;
    $testprefix .= "orph_";
}

# Remove remaining marks for the orpahn regions and its variables:
($code) = enlarge_tags('ompts:orphan','','',($code));
($code) = enlarge_tags('ompts:orphan:vars','','',($code));

if($test) {
# Remove the marks for the testcode and remove the code for the crosstests: 
    ($code) = enlarge_tags('ompts:check','','',($code));
    ($code) = delete_tags('ompts:crosscheck',($code));		
    $testprefix .= "test_";
}
elsif($crosstest) {
# Remove the marks for the crosstestcode and remove the code for the tests: 
    ($code) = enlarge_tags('ompts:crosscheck','','',($code));
    ($code) = delete_tags('ompts:check',($code));		
    $testprefix .= "ctest_";
}
# Making some final modifications:
($code) = replace_tags('testfunctionname',$testprefix.$functionname,($code));
($code) = replace_tags('directive',$directive,($code));
($code) = replace_tags('description',$description,($code));
($code) = enlarge_tags('ompts:testcode:functionname',$testprefix,'',($code) );
#	$code =  "\#include \"omp_testsuite.h\"\n".$code;
# Write the result into the file and close it:
print OUTFILE $code;
close(OUTFILE);
