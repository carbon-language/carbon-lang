#!/usr/bin/perl -w
#
# Program:  profile.pl
#
# Synopsis: Insert instrumentation code into a program, run it with the JIT,
#           then print out a profile report.
#
# Syntax:   profile.pl [OPTIONS] bytecodefile <arguments>
#
# OPTIONS may include one or more of the following:
#     -block - Enable basic block level profiling
#
# Any unrecognized options are passed into the invocation of llvm-prof
#

my $ProfilePass = "-insert-function-profiling";

my $LLVMProfOpts = "";

# Parse arguments...
while (scalar(@ARGV) and ($_ = $ARGV[0], /^[-+]/)) {
  shift;
  last if /^--$/;  # Stop processing arguments on --

  # List command line options here...
  if (/^-?-block$/) { $ProfilePass = "-insert-block-profiling"; next; }
  if (/^-?-help$/) {
    print "OVERVIEW: profile.pl - Instrumentation and profile printer.\n\n";
    print "USAGE: profile.pl [options] program.bc <program args>\n\n";
    print "OPTIONS:\n";
    print "  -block - Enable basic block level profiling\n";
    print "  -help  - Print this usage information\n";
    print "\nAll other options are passed into llvm-prof.\n";
    exit 1;
  }

  # Otherwise, pass the option on to llvm-prof
  $LLVMProfOpts .= " " . $_;
}

die "Must specify LLVM bytecode file as first argument!" if (@ARGV == 0);

my $BytecodeFile = $ARGV[0];

shift @ARGV;

my $LLIPath = `which lli`;
$LLIPath = `dirname $LLIPath`;
chomp $LLIPath;

my $LibProfPath = $LLIPath . "/../../lib/Debug/libprofile_rt.so";

system "opt -q $ProfilePass < $BytecodeFile | lli -fake-argv0 '$BytecodeFile'" .
       " -load $LibProfPath - " . (join ' ', @ARGV);

system "llvm-prof $LLVMProfOpts $BytecodeFile";
