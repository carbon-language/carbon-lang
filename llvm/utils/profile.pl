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
#     -block    - Enable basicblock-level profiling
#     -function - Enable function-level profiling
#     -o <filename> - Emit profiling information to the specified file, instead
#                     of llvmprof.out
#
# Any unrecognized options are passed into the invocation of llvm-prof
#

my $ProfilePass = "-insert-block-profiling";

my $LLVMProfOpts = "";
my $ProgramOpts = "";
my $ProfileFile = "";

# Parse arguments...
while (scalar(@ARGV) and ($_ = $ARGV[0], /^[-+]/)) {
  shift;
  last if /^--$/;  # Stop processing arguments on --

  # List command line options here...
  if (/^-?-block$/) { $ProfilePass = "-insert-block-profiling"; next; }
  if (/^-?-function$/) { $ProfilePass = "-insert-function-profiling"; next; }
  if (/^-?-o$/) {         # Read -o filename...
    die "-o option requires a filename argument!" if (!scalar(@ARGV));
    $ProgramOpts .= " -llvmprof-output $ARGV[0]";
    $ProfileFile = $ARGV[0];
    shift;
    next;
  }
  if (/^-?-help$/) {
    print "OVERVIEW: profile.pl - Instrumentation and profile printer.\n\n";
    print "USAGE: profile.pl [options] program.bc <program args>\n\n";
    print "OPTIONS:\n";
    print "  -block    - Enable basicblock-level profiling\n";
    print "  -function - Enable function-level profiling\n";
    print "  -o <file> - Specify an output file other than llvm-prof.out.\n";
    print "  -help     - Print this usage information\n";
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
       " -load $LibProfPath -$ProgramOpts " . (join ' ', @ARGV);

system "llvm-prof $LLVMProfOpts $BytecodeFile $ProfileFile";
