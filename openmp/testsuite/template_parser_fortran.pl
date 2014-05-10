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
my $mainprocsrc = "ompts_standaloneProc.f"; 

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
($description)  = get_tag_values ('ompts:testdescription',$src);
($directive)    = get_tag_values ('ompts:directive',$src);
($functionname) = get_tag_values ('ompts:testcode:functionname',$src);

open (OUTFILE,">$outfile") or die "Could not create the output file for $directive";

# Creating the source for the test:
($code) = get_tag_values('ompts:testcode',$src);
# Putting together the functions and the mainprogramm:
$code .= $mainproc;

#thanks to Dr. Yin Ma in Absoft, get the parameters <ompts:orphan:params> by joon
 ($parms) = get_tag_values('ompts:orphan:parms',($code));
 ($parms) = leave_single_space($parms);
 ($code) = replace_tags('ompts:orphan:parms','',$code);

# Make modifications for the orphaned testversion if necessary:
if ($orphan) {
# Get the global variables:
    @defs = get_tag_values("ompts:orphan:vars",$code);
    $orphvarsdef = "";
    foreach (@defs) {
        if (not /^[ ]*$/gs) { $orphvarsdef = join("\n",$orphvarsdef,$_); } 
    }
# Generate the orphan subroutines:
        $orphfuncs = create_orph_fortranfunctions ("", ($code),($parms));
# Replace orphan regions by functioncalls:
        ($code) = orphan_regions2fortranfunctions ("", ($code),($parms));
        ($code) = enlarge_tags ('ompts:orphan:vars','','',($code));
# to find orphan call statement and add parameters, by joon
        ($code) = enlarge_tags('ompts:orphan:parms','','',($code));
# Put all together:
        $code = $code . $orphfuncs;
}

# Remove remaining marks for the orpahn regions and its variables:
($code) = enlarge_tags('ompts:orphan','','',($code));
($code) = enlarge_tags('ompts:orphan:vars','','',($code));
# remove parameters between for orphaned directive parametes, added byjoon
($code) = enlarge_tags('ompts:orphan:parms','','',($code));

if($test) {
# Remove the marks for the testcode and remove the code for the crosstests: 
    ($code) = enlarge_tags('ompts:check','','',($code));
    ($code) = delete_tags('ompts:crosscheck',($code));		
}
elsif($crosstest) {
# Remove the marks for the crosstestcode and remove the code for the tests: 
    ($code) = enlarge_tags('ompts:crosscheck','','',($code));
    ($code) = delete_tags('ompts:check',($code));		
}
# Making some final modifications:
($code) = replace_tags('testfunctionname',"test_".$functionname,($code));
($code) = replace_tags('directive',$directive,($code));
($code) = replace_tags('description',$description,($code));
($code) = enlarge_tags('ompts:testcode:functionname',"test_",'',($code) );
#	$code =  "\#include \"omp_testsuite.h\"\n".$code;
# Write the result into the file and close it:
print OUTFILE $code;
close(OUTFILE);
