#!/usr/bin/perl
use POSIX qw(strftime);
use File::Copy;
use File::Find;
use Socket;

#
# Program:  NewNightlyTest.pl
#
# Synopsis: Perform a series of tests which are designed to be run nightly.
#           This is used to keep track of the status of the LLVM tree, tracking
#           regressions and performance changes. Submits this information
#           to llvm.org where it is placed into the nightlytestresults database.
#
# Syntax:   NightlyTest.pl [OPTIONS] [CVSROOT BUILDDIR WEBDIR]
#   where
# OPTIONS may include one or more of the following:
#
# MAIN OPTIONS:
#  -config LLVMPATH If specified, use an existing LLVM build and only run and
#                   report the test information. The LLVMCONFIG argument should
#                   be the path to the llvm-config executable in the LLVM build.
#                   This should be the first argument if given. NOT YET
#                   IMPLEMENTED.
#  -nickname NAME   The NAME argument specifieds the nickname this script
#                   will submit to the nightlytest results repository.
#  -submit-server   Specifies a server to submit the test results too. If this
#                   option is not specified it defaults to
#                   llvm.org. This is basically just the address of the
#                   webserver
#  -submit-script   Specifies which script to call on the submit server. If
#                   this option is not specified it defaults to
#                   /nightlytest/NightlyTestAccept.php. This is basically
#                   everything after the www.yourserver.org.
#  -submit-aux      If specified, an auxiliary script to run in addition to the
#                   normal submit script. The script will be passed the path to
#                   the "sentdata.txt" file as its sole argument.
#  -nosubmit        Do not report the test results back to a submit server.
#
#
# BUILD OPTIONS (not used with -config):
#  -nocheckout      Do not create, checkout, update, or configure
#                   the source tree.
#  -noremove        Do not remove the BUILDDIR after it has been built.
#  -noremoveresults Do not remove the WEBDIR after it has been built.
#  -nobuild         Do not build llvm. If tests are enabled perform them
#                   on the llvm build specified in the build directory
#  -release         Build an LLVM Release version
#  -release-asserts Build an LLVM ReleaseAsserts version
#  -disable-bindings     Disable building LLVM bindings.
#  -with-clang      Checkout Clang source into tools/clang.
#  -compileflags    Next argument specifies extra options passed to make when
#                   building LLVM.
#  -use-gmake       Use gmake instead of the default make command to build
#                   llvm and run tests.
#
# TESTING OPTIONS:
#  -notest          Do not even attempt to run the test programs.
#  -nodejagnu       Do not run feature or regression tests
#  -enable-llcbeta  Enable testing of beta features in llc.
#  -enable-lli      Enable testing of lli (interpreter) features, default is off
#  -disable-pic	    Disable building with Position Independent Code.
#  -disable-llc     Disable LLC tests in the nightly tester.
#  -disable-jit     Disable JIT tests in the nightly tester.
#  -disable-cbe     Disable C backend tests in the nightly tester.
#  -disable-lto     Disable link time optimization.
#  -test-cflags     Next argument specifies that C compilation options that
#                   override the default when running the testsuite.
#  -test-cxxflags   Next argument specifies that C++ compilation options that
#                   override the default when running the testsuite.
#  -extraflags      Next argument specifies extra options that are passed to
#                   compile the tests.
#  -noexternals     Do not run the external tests (for cases where povray
#                   or SPEC are not installed)
#  -with-externals  Specify a directory where the external tests are located.
#
# OTHER OPTIONS:
#  -parallel        Run parallel jobs with GNU Make (see -parallel-jobs).
#  -parallel-jobs   The number of parallel Make jobs to use (default is two).
#  -verbose         Turn on some debug output
#  -nice            Checkout/Configure/Build with "nice" to reduce impact
#                   on busy servers.
#  -f2c             Next argument specifies path to F2C utility
#  -gccpath         Path to gcc/g++ used to build LLVM
#  -target          Specify the target triplet
#  -cflags          Next argument specifies that C compilation options that
#                   override the default.
#  -cxxflags        Next argument specifies that C++ compilation options that
#                   override the default.
#  -ldflags         Next argument specifies that linker options that override
#                   the default.
#
# CVSROOT is ignored, it is passed for backwards compatibility.
# BUILDDIR is the directory where sources for this test run will be checked out
#  AND objects for this test run will be built. This directory MUST NOT
#  exist before the script is run; it will be created by the svn checkout
#  process and erased (unless -noremove is specified; see above.)
# WEBDIR is the directory into which the test results web page will be written,
#  AND in which the "index.html" is assumed to be a symlink to the most recent
#  copy of the results. This directory will be created if it does not exist.
# LLVMGCCDIR is the directory in which the LLVM GCC Front End is installed
#  to. This is the same as you would have for a normal LLVM build.
#
##############################################################
#
# Getting environment variables
#
##############################################################
my $HOME       = $ENV{'HOME'};
my $SVNURL     = $ENV{"SVNURL"};
$SVNURL        = 'http://llvm.org/svn/llvm-project' unless $SVNURL;
my $TestSVNURL = $ENV{"TestSVNURL"};
$TestSVNURL    = 'http://llvm.org/svn/llvm-project' unless $TestSVNURL;
my $BuildDir   = $ENV{'BUILDDIR'};
my $WebDir     = $ENV{'WEBDIR'};

my $LLVMSrcDir   = $ENV{'LLVMSRCDIR'};
$LLVMSrcDir    = "$BuildDir/llvm" unless $LLVMSrcDir;
my $LLVMObjDir   = $ENV{'LLVMOBJDIR'};
$LLVMObjDir    = "$BuildDir/llvm" unless $LLVMObjDir;
my $LLVMTestDir   = $ENV{'LLVMTESTDIR'};
$LLVMTestDir    = "$BuildDir/llvm/projects/llvm-test" unless $LLVMTestDir;

##############################################################
#
# Calculate the date prefix...
#
##############################################################
@TIME = localtime;
my $DATE = sprintf "%4d-%02d-%02d_%02d-%02d", $TIME[5]+1900, $TIME[4]+1, $TIME[3], $TIME[1], $TIME[0];

##############################################################
#
# Parse arguments...
#
##############################################################
$CONFIG_PATH="";
$CONFIGUREARGS="";
$nickname="";
$NOTEST=0;
$MAKECMD="make";
$SUBMITSERVER = "llvm.org";
$SUBMITSCRIPT = "/nightlytest/NightlyTestAccept.php";
$SUBMITAUX="";
$SUBMIT = 1;
$PARALLELJOBS = "2";
my $TESTFLAGS="";

while (scalar(@ARGV) and ($_ = $ARGV[0], /^[-+]/)) {
  shift;
  last if /^--$/;  # Stop processing arguments on --

  # List command line options here...
  if (/^-config$/)         { $CONFIG_PATH = "$ARGV[0]"; shift; next; }
  if (/^-nocheckout$/)     { $NOCHECKOUT = 1; next; }
  if (/^-noremove$/)       { $NOREMOVE = 1; next; }
  if (/^-noremoveatend$/)  { $NOREMOVEATEND = 1; next; }
  if (/^-noremoveresults$/){ $NOREMOVERESULTS = 1; next; }
  if (/^-notest$/)         { $NOTEST = 1; next; }
  if (/^-norunningtests$/) { next; } # Backward compatibility, ignored.
  if (/^-parallel-jobs$/)  { $PARALLELJOBS = "$ARGV[0]"; shift; next;}
  if (/^-parallel$/)       { $MAKEOPTS = "$MAKEOPTS -j$PARALLELJOBS -l3.0"; next; }
  if (/^-with-clang$/)     { $WITHCLANG = 1; next; }
  if (/^-release$/)        { $MAKEOPTS = "$MAKEOPTS ENABLE_OPTIMIZED=1 ".
                             "OPTIMIZE_OPTION=-O2"; $BUILDTYPE="release"; next;}
  if (/^-release-asserts$/){ $MAKEOPTS = "$MAKEOPTS ENABLE_OPTIMIZED=1 ".
                             "DISABLE_ASSERTIONS=1 ".
                             "OPTIMIZE_OPTION=-O2";
                             $BUILDTYPE="release-asserts"; next;}
  if (/^-enable-llcbeta$/) { $PROGTESTOPTS .= " ENABLE_LLCBETA=1"; next; }
  if (/^-disable-pic$/)    { $CONFIGUREARGS .= " --enable-pic=no"; next; }
  if (/^-enable-lli$/)     { $PROGTESTOPTS .= " ENABLE_LLI=1";
                             $CONFIGUREARGS .= " --enable-lli"; next; }
  if (/^-disable-llc$/)    { $PROGTESTOPTS .= " DISABLE_LLC=1";
                             $CONFIGUREARGS .= " --disable-llc_diffs"; next; }
  if (/^-disable-jit$/)    { $PROGTESTOPTS .= " DISABLE_JIT=1";
                             $CONFIGUREARGS .= " --disable-jit"; next; }
  if (/^-disable-bindings$/)    { $CONFIGUREARGS .= " --disable-bindings"; next; }
  if (/^-disable-cbe$/)    { $PROGTESTOPTS .= " DISABLE_CBE=1"; next; }
  if (/^-disable-lto$/)    { $PROGTESTOPTS .= " DISABLE_LTO=1"; next; }
  if (/^-test-opts$/)      { $PROGTESTOPTS .= " $ARGV[0]"; shift; next; }
  if (/^-verbose$/)        { $VERBOSE = 1; next; }
  if (/^-teelogs$/)        { $TEELOGS = 1; next; }
  if (/^-nice$/)           { $NICE = "nice "; next; }
  if (/^-f2c$/)            { $CONFIGUREARGS .= " --with-f2c=$ARGV[0]";
                             shift; next; }
  if (/^-with-externals$/) { $CONFIGUREARGS .= " --with-externals=$ARGV[0]";
                             shift; next; }
  if (/^-configure-args$/) { $CONFIGUREARGS .= " $ARGV[0]";
                             shift; next; }
  if (/^-submit-server/)   { $SUBMITSERVER = "$ARGV[0]"; shift; next; }
  if (/^-submit-script/)   { $SUBMITSCRIPT = "$ARGV[0]"; shift; next; }
  if (/^-submit-aux/)      { $SUBMITAUX = "$ARGV[0]"; shift; next; }
  if (/^-nosubmit$/)       { $SUBMIT = 0; next; }
  if (/^-nickname$/)       { $nickname = "$ARGV[0]"; shift; next; }
  if (/^-gccpath/)         { $CONFIGUREARGS .=
                             " CC=$ARGV[0]/gcc CXX=$ARGV[0]/g++";
                             $GCCPATH=$ARGV[0]; shift;  next; }
  else                     { $GCCPATH=""; }
  if (/^-target/)          { $CONFIGUREARGS .= " --target=$ARGV[0]";
                             shift; next; }
  if (/^-cflags/)          { $MAKEOPTS = "$MAKEOPTS C.Flags=\'$ARGV[0]\'";
                             shift; next; }
  if (/^-cxxflags/)        { $MAKEOPTS = "$MAKEOPTS CXX.Flags=\'$ARGV[0]\'";
                             shift; next; }
  if (/^-ldflags/)         { $MAKEOPTS = "$MAKEOPTS LD.Flags=\'$ARGV[0]\'";
                             shift; next; }
  if (/^-test-cflags/)     { $TESTFLAGS = "$TESTFLAGS CFLAGS=\'$ARGV[0]\'";
                             shift; next; }
  if (/^-test-cxxflags/)   { $TESTFLAGS = "$TESTFLAGS CXXFLAGS=\'$ARGV[0]\'";
                             shift; next; }
  if (/^-compileflags/)    { $MAKEOPTS = "$MAKEOPTS $ARGV[0]"; shift; next; }
  if (/^-use-gmake/)       { $MAKECMD = "gmake"; shift; next; }
  if (/^-extraflags/)      { $CONFIGUREARGS .=
                             " --with-extra-options=\'$ARGV[0]\'"; shift; next;}
  if (/^-noexternals$/)    { $NOEXTERNALS = 1; next; }
  if (/^-nodejagnu$/)      { $NODEJAGNU = 1; next; }
  if (/^-nobuild$/)        { $NOBUILD = 1; next; }
  print "Unknown option: $_ : ignoring!\n";
}

if ($ENV{'LLVMGCCDIR'}) {
  $CONFIGUREARGS .= " --with-llvmgccdir=" . $ENV{'LLVMGCCDIR'};
  $LLVMGCCPATH = $ENV{'LLVMGCCDIR'} . '/bin';
}
else {
  $LLVMGCCPATH = "";
}

if ($CONFIGUREARGS !~ /--disable-jit/) {
  $CONFIGUREARGS .= " --enable-jit";
}

if (@ARGV != 0 and @ARGV != 3) {
  die "error: must specify 0 or 3 options!";
}

if (@ARGV == 3) {
  if ($CONFIG_PATH ne "") {
      die "error: arguments are unsupported in -config mode,";
  }

  # ARGV[0] used to be the CVS root, ignored for backward compatibility.
  $BuildDir   = $ARGV[1];
  $WebDir     = $ARGV[2];
}

if ($BuildDir   eq "" or
    $WebDir     eq "") {
  die("please specify a build directory, and a web directory");
 }

if ($nickname eq "") {
  die ("Please invoke NewNightlyTest.pl with command line option " .
       "\"-nickname <nickname>\"");
}

if ($BUILDTYPE ne "release" && $BUILDTYPE ne "release-asserts") {
  $BUILDTYPE = "debug";
}

if ($CONFIG_PATH ne "") {
  die "error: -config mode is not yet implemented,";
}

##############################################################
#
# Define the file names we'll use
#
##############################################################
my $Prefix = "$WebDir/$DATE";
my $BuildLog = "$Prefix-Build-Log.txt";
my $COLog = "$Prefix-CVS-Log.txt";
my $SingleSourceLog = "$Prefix-SingleSource-ProgramTest.txt.gz";
my $MultiSourceLog = "$Prefix-MultiSource-ProgramTest.txt.gz";
my $ExternalLog = "$Prefix-External-ProgramTest.txt.gz";
my $DejagnuLog = "$Prefix-Dejagnu-testrun.log";
my $DejagnuSum = "$Prefix-Dejagnu-testrun.sum";
my $DejagnuTestsLog = "$Prefix-DejagnuTests-Log.txt";
if (! -d $WebDir) {
  mkdir $WebDir, 0777 or die "Unable to create web directory: '$WebDir'.";
  if($VERBOSE){
    warn "$WebDir did not exist; creating it.\n";
  }
}

if ($VERBOSE) {
  print "INITIALIZED\n";
  print "SVN URL  = $SVNURL\n";
  print "COLog    = $COLog\n";
  print "BuildDir = $BuildDir\n";
  print "WebDir   = $WebDir\n";
  print "Prefix   = $Prefix\n";
  print "BuildLog = $BuildLog\n";
}

##############################################################
#
# Helper functions
#
##############################################################

sub GetDir {
  my $Suffix = shift;
  opendir DH, $WebDir;
  my @Result = reverse sort grep !/$DATE/, grep /[-0-9]+$Suffix/, readdir DH;
  closedir DH;
  return @Result;
}

sub RunLoggedCommand {
  my $Command = shift;
  my $Log = shift;
  my $Title = shift;
  if ($TEELOGS) {
      if ($VERBOSE) {
          print "$Title\n";
          print "$Command 2>&1 | tee $Log\n";
      }
      system "$Command 2>&1 | tee $Log";
  } else {
      if ($VERBOSE) {
          print "$Title\n";
          print "$Command 2>&1 > $Log\n";
      }
      system "$Command 2>&1 > $Log";
  }
}

sub RunAppendingLoggedCommand {
  my $Command = shift;
  my $Log = shift;
  my $Title = shift;
  if ($TEELOGS) {
      if ($VERBOSE) {
          print "$Title\n";
          print "$Command 2>&1 | tee -a $Log\n";
      }
      system "$Command 2>&1 | tee -a $Log";
  } else {
      if ($VERBOSE) {
          print "$Title\n";
          print "$Command 2>&1 > $Log\n";
      }
      system "$Command 2>&1 >> $Log";
  }
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub GetRegex {   # (Regex with ()'s, value)
  $_[1] =~ /$_[0]/m;
  return $1
    if (defined($1));
  return "0";
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub GetRegexNum {
  my ($Regex, $Num, $Regex2, $File) = @_;
  my @Items = split "\n", `grep '$Regex' $File`;
  return GetRegex $Regex2, $Items[$Num];
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub ChangeDir { # directory, logical name
  my ($dir,$name) = @_;
  chomp($dir);
  if ( $VERBOSE ) { print "Changing To: $name ($dir)\n"; }
  $result = chdir($dir);
  if (!$result) {
    print "ERROR!!! Cannot change directory to: $name ($dir) because $!";
    return false;
  }
  return true;
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub ReadFile {
  if (open (FILE, $_[0])) {
    undef $/;
    my $Ret = <FILE>;
    close FILE;
    $/ = '\n';
    return $Ret;
  } else {
    print "Could not open file '$_[0]' for reading!\n";
    return "";
  }
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub WriteFile {  # (filename, contents)
  open (FILE, ">$_[0]") or die "Could not open file '$_[0]' for writing!\n";
  print FILE $_[1];
  close FILE;
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub CopyFile { #filename, newfile
  my ($file, $newfile) = @_;
  chomp($file);
  if ($VERBOSE) { print "Copying $file to $newfile\n"; }
  copy($file, $newfile);
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub AddRecord {
  my ($Val, $Filename,$WebDir) = @_;
  my @Records;
  if (open FILE, "$WebDir/$Filename") {
    @Records = grep !/$DATE/, split "\n", <FILE>;
    close FILE;
  }
  push @Records, "$DATE: $Val";
  WriteFile "$WebDir/$Filename", (join "\n", @Records) . "\n";
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# FormatTime - Convert a time from 1m23.45 into 83.45
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub FormatTime {
  my $Time = shift;
  if ($Time =~ m/([0-9]+)m([0-9.]+)/) {
    $Time = sprintf("%7.4f", $1*60.0+$2);
  }
  return $Time;
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This function is meant to read in the dejagnu sum file and
# return a string with only the results (i.e. PASS/FAIL/XPASS/
# XFAIL).
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub GetDejagnuTestResults { # (filename, log)
    my ($filename, $DejagnuLog) = @_;
    my @lines;
    $/ = "\n"; #Make sure we're going line at a time.

    if( $VERBOSE) { print "DEJAGNU TEST RESULTS:\n"; }

    if (open SRCHFILE, $filename) {
        # Process test results
        while ( <SRCHFILE> ) {
            if ( length($_) > 1 ) {
                chomp($_);
                if ( m/^(PASS|XPASS|FAIL|XFAIL): .*\/llvm\/test\/(.*)$/ ) {
                    push(@lines, "$1: test/$2");
                }
            }
        }
    }
    close SRCHFILE;

    my $content = join("\n", @lines);
    return $content;
}



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This function acts as a mini web browswer submitting data
# to our central server via the post method
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub SendData {
    $host = $_[0];
    $file = $_[1];
    $variables = $_[2];

    # Write out the "...-sentdata.txt" file.

    my $sentdata="";
    foreach $x (keys (%$variables)){
        $value = $variables->{$x};
        $sentdata.= "$x  => $value\n";
    }
    WriteFile "$Prefix-sentdata.txt", $sentdata;

    if (!($SUBMITAUX eq "")) {
        system "$SUBMITAUX \"$Prefix-sentdata.txt\"";
    }

    if (!$SUBMIT) {
        return "Skipped standard submit.\n";
    }

    # Create the content to send to the server.

    my $content;
    foreach $key (keys (%$variables)){
        $value = $variables->{$key};
        $value =~ s/([^A-Za-z0-9])/sprintf("%%%02X", ord($1))/seg;
        $content .= "$key=$value&";
    }

    # Send the data to the server.
    #
    # FIXME: This code should be more robust?

    $port=80;
    $socketaddr= sockaddr_in $port, inet_aton $host or die "Bad hostname\n";
    socket SOCK, PF_INET, SOCK_STREAM, getprotobyname('tcp') or
      die "Bad socket\n";
    connect SOCK, $socketaddr or die "Bad connection\n";
    select((select(SOCK), $| = 1)[0]);

    $length = length($content);

    my $send= "POST $file HTTP/1.0\n";
    $send.= "Host: $host\n";
    $send.= "Content-Type: application/x-www-form-urlencoded\n";
    $send.= "Content-length: $length\n\n";
    $send.= "$content";

    print SOCK $send;
    my $result;
    while(<SOCK>){
        $result  .= $_;
    }
    close(SOCK);

    return $result;
}

##############################################################
#
# Getting Start timestamp
#
##############################################################
$starttime = `date "+20%y-%m-%d %H:%M:%S"`;

##############################################################
#
# Create the source repository directory
#
##############################################################
if (!$NOCHECKOUT) {
  if (-d $BuildDir) {
    if (!$NOREMOVE) {
      if ( $VERBOSE ) {
        print "Build directory exists! Removing it\n";
      }
      system "rm -rf $BuildDir";
      mkdir $BuildDir or die "Could not create checkout directory $BuildDir!";
    } else {
      if ( $VERBOSE ) {
        print "Build directory exists!\n";
      }
    }
  } else {
    mkdir $BuildDir or die "Could not create checkout directory $BuildDir!";
  }
}


##############################################################
#
# Check out the llvm tree with SVN
#
##############################################################
if (!$NOCHECKOUT) {
  ChangeDir( $BuildDir, "checkout directory" );
  my $SVNCMD = "$NICE svn co --non-interactive $SVNURL";
  my $SVNCMD2 = "$NICE svn co --non-interactive $TestSVNURL";
  RunLoggedCommand("( time -p $SVNCMD/llvm/trunk llvm; cd llvm/projects ; " .
                   "$SVNCMD2/test-suite/trunk llvm-test )", $COLog,
                   "CHECKOUT LLVM");
  if ($WITHCLANG) {
      my $SVNCMD = "$NICE svn co --non-interactive $SVNURL/cfe/trunk";
      RunLoggedCommand("( time -p cd llvm/tools ; $SVNCMD clang )", $COLog,
                       "CHECKOUT CLANG");
  }
}
ChangeDir( $LLVMSrcDir , "llvm source directory") ;

##############################################################
#
# Get some static statistics about the current source code
#
# This can probably be put on the server side
#
##############################################################
my $CheckoutTime_Wall = GetRegex "([0-9.]+)", `grep '^real' $COLog`;
my $CheckoutTime_User = GetRegex "([0-9.]+)", `grep '^user' $COLog`;
my $CheckoutTime_Sys = GetRegex "([0-9.]+)", `grep '^sys' $COLog`;
my $CheckoutTime_CPU = $CVSCheckoutTime_User + $CVSCheckoutTime_Sys;

##############################################################
#
# Build the entire tree, saving build messages to the build log
#
##############################################################
if (!$NOCHECKOUT && !$NOBUILD) {
  my $EXTRAFLAGS = "--enable-spec --with-objroot=.";
  RunLoggedCommand("(time -p $NICE ./configure $CONFIGUREARGS $EXTRAFLAGS) ",
                   $BuildLog, "CONFIGURE");
  # Build the entire tree, capturing the output into $BuildLog
  RunAppendingLoggedCommand("(time -p $NICE $MAKECMD clean)", $BuildLog, "BUILD CLEAN");
  RunAppendingLoggedCommand("(time -p $NICE $MAKECMD $MAKEOPTS)", $BuildLog, "BUILD");
}

##############################################################
#
# Get some statistics about the build...
#
##############################################################

# Get the time taken by the configure script
my $ConfigTimeU = GetRegexNum "^user", 0, "([0-9.]+)", "$BuildLog";
my $ConfigTimeS = GetRegexNum "^sys", 0, "([0-9.]+)", "$BuildLog";
my $ConfigTime  = $ConfigTimeU+$ConfigTimeS;  # ConfigTime = User+System
my $ConfigWallTime = GetRegexNum "^real", 0,"([0-9.]+)","$BuildLog";

$ConfigTime=-1 unless $ConfigTime;
$ConfigWallTime=-1 unless $ConfigWallTime;

my $BuildTimeU = GetRegexNum "^user", 1, "([0-9.]+)", "$BuildLog";
my $BuildTimeS = GetRegexNum "^sys", 1, "([0-9.]+)", "$BuildLog";
my $BuildTime  = $BuildTimeU+$BuildTimeS;  # BuildTime = User+System
my $BuildWallTime = GetRegexNum "^real", 1, "([0-9.]+)","$BuildLog";

$BuildTime=-1 unless $BuildTime;
$BuildWallTime=-1 unless $BuildWallTime;

my $BuildError = 0, $BuildStatus = "OK";
if ($NOBUILD) {
  $BuildStatus = "Skipped by user";
}
elsif (`grep '^$MAKECMD\[^:]*: .*Error' $BuildLog | wc -l` + 0 ||
  `grep '^$MAKECMD: \*\*\*.*Stop.' $BuildLog | wc -l`+0) {
  $BuildStatus = "Error: compilation aborted";
  $BuildError = 1;
  if( $VERBOSE) { print  "\n***ERROR BUILDING TREE\n\n"; }
}
if ($BuildError) { $NODEJAGNU=1; }

##############################################################
#
# Running dejagnu tests
#
##############################################################
my $DejangnuTestResults=""; # String containing the results of the dejagnu
my $dejagnu_output = "$DejagnuTestsLog";
if (!$NODEJAGNU) {
  #Run the feature and regression tests, results are put into testrun.sum
  #Full log in testrun.log
  RunLoggedCommand("(time -p $MAKECMD $MAKEOPTS check)", $dejagnu_output, "DEJAGNU");

  #Copy the testrun.log and testrun.sum to our webdir
  CopyFile("test/testrun.log", $DejagnuLog);
  CopyFile("test/testrun.sum", $DejagnuSum);
  #can be done on server
  $DejagnuTestResults = GetDejagnuTestResults($DejagnuSum, $DejagnuLog);
  $unexpfail_tests = $DejagnuTestResults;
}

#Extract time of dejagnu tests
my $DejagnuTimeU = GetRegexNum "^user", 0, "([0-9.]+)", "$dejagnu_output";
my $DejagnuTimeS = GetRegexNum "^sys", 0, "([0-9.]+)", "$dejagnu_output";
$DejagnuTime  = $DejagnuTimeU+$DejagnuTimeS;  # DejagnuTime = User+System
$DejagnuWallTime = GetRegexNum "^real", 0,"([0-9.]+)","$dejagnu_output";
$DejagnuTestResults =
  "Dejagnu skipped by user choice." unless $DejagnuTestResults;
$DejagnuTime     = "0.0" unless $DejagnuTime;
$DejagnuWallTime = "0.0" unless $DejagnuWallTime;

if (!$NODEJAGNU) {
  if ( $VERBOSE ) { print "BUILD INFORMATION COLLECTION STAGE\n"; }
} #endif !NODEGAGNU

##############################################################
#
# If we built the tree successfully, run the nightly programs tests...
#
# A set of tests to run is passed in (i.e. "SingleSource" "MultiSource"
# "External")
#
##############################################################

sub TestDirectory {
  my $SubDir = shift;
  ChangeDir( "$LLVMTestDir/$SubDir",
             "Programs Test Subdirectory" ) || return ("", "");

  my $ProgramTestLog = "$Prefix-$SubDir-ProgramTest.txt";

  # Run the programs tests... creating a report.nightly.csv file
  if (!$NOTEST) {
    if( $VERBOSE) {
      print "$MAKECMD -k $MAKEOPTS $PROGTESTOPTS report.nightly.csv ".
            "$TESTFLAGS TEST=nightly > $ProgramTestLog 2>&1\n";
    }
    RunLoggedCommand("$MAKECMD -k $MAKEOPTS $PROGTESTOPTS report.nightly.csv ".
                     "$TESTFLAGS TEST=nightly",
                     $ProgramTestLog, "TEST DIRECTORY $SubDir");
    $llcbeta_options=`$MAKECMD print-llcbeta-option`;
  }

  my $ProgramsTable;
  if (`grep '^$MAKECMD\[^:]: .*Error' $ProgramTestLog | wc -l` + 0) {
    $TestError = 1;
    $ProgramsTable="Error running test $SubDir\n";
    print "ERROR TESTING\n";
  } elsif (`grep '^$MAKECMD\[^:]: .*No rule to make target' $ProgramTestLog | wc -l` + 0) {
    $TestError = 1;
    $ProgramsTable="Makefile error running tests $SubDir!\n";
    print "ERROR TESTING\n";
  } else {
    $TestError = 0;
  #
  # Create a list of the tests which were run...
  #
  system "egrep 'TEST-(PASS|FAIL)' < $ProgramTestLog ".
         "| sort > $Prefix-$SubDir-Tests.txt";
  }
  $ProgramsTable = ReadFile "report.nightly.csv";

  ChangeDir( "../../..", "Programs Test Parent Directory" );
  return ($ProgramsTable, $llcbeta_options);
} #end sub TestDirectory

##############################################################
#
# Calling sub TestDirectory
#
##############################################################
if (!$BuildError) {
  ($SingleSourceProgramsTable, $llcbeta_options) =
    TestDirectory("SingleSource");
  WriteFile "$Prefix-SingleSource-Performance.txt", $SingleSourceProgramsTable;
  ($MultiSourceProgramsTable, $llcbeta_options) = TestDirectory("MultiSource");
  WriteFile "$Prefix-MultiSource-Performance.txt", $MultiSourceProgramsTable;
  if ( ! $NOEXTERNALS ) {
    ($ExternalProgramsTable, $llcbeta_options) = TestDirectory("External");
    WriteFile "$Prefix-External-Performance.txt", $ExternalProgramsTable;
    system "cat $Prefix-SingleSource-Tests.txt " .
               "$Prefix-MultiSource-Tests.txt ".
               "$Prefix-External-Tests.txt | sort > $Prefix-Tests.txt";
    system "cat $Prefix-SingleSource-Performance.txt " .
               "$Prefix-MultiSource-Performance.txt ".
               "$Prefix-External-Performance.txt | sort > $Prefix-Performance.txt";
  } else {
    $ExternalProgramsTable = "External TEST STAGE SKIPPED\n";
    if ( $VERBOSE ) {
      print "External TEST STAGE SKIPPED\n";
    }
    system "cat $Prefix-SingleSource-Tests.txt " .
               "$Prefix-MultiSource-Tests.txt ".
               " | sort > $Prefix-Tests.txt";
    system "cat $Prefix-SingleSource-Performance.txt " .
               "$Prefix-MultiSource-Performance.txt ".
               " | sort > $Prefix-Performance.txt";
  }

  ##############################################################
  #
  #
  # gathering tests added removed broken information here
  #
  #
  ##############################################################
  my $dejagnu_test_list = ReadFile "$Prefix-Tests.txt";
  my @DEJAGNU = split "\n", $dejagnu_test_list;
  my ($passes, $fails, $xfails) = "";

  if(!$NODEJAGNU) {
    for ($x=0; $x<@DEJAGNU; $x++) {
      if ($DEJAGNU[$x] =~ m/^PASS:/) {
        $passes.="$DEJAGNU[$x]\n";
      }
      elsif ($DEJAGNU[$x] =~ m/^FAIL:/) {
        $fails.="$DEJAGNU[$x]\n";
      }
      elsif ($DEJAGNU[$x] =~ m/^XFAIL:/) {
        $xfails.="$DEJAGNU[$x]\n";
      }
    }
  }

} #end if !$BuildError

##############################################################
#
# Getting end timestamp
#
##############################################################
$endtime = `date "+20%y-%m-%d %H:%M:%S"`;


##############################################################
#
# Place all the logs neatly into one humungous file
#
##############################################################
if ( $VERBOSE ) { print "PREPARING LOGS TO BE SENT TO SERVER\n"; }

$machine_data = "uname: ".`uname -a`.
                "hardware: ".`uname -m`.
                "os: ".`uname -sr`.
                "name: ".`uname -n`.
                "date: ".`date \"+20%y-%m-%d\"`.
                "time: ".`date +\"%H:%M:%S\"`;

my @CVS_DATA;
my $cvs_data;
@CVS_DATA = ReadFile "$COLog";
$cvs_data = join("\n", @CVS_DATA);

my @BUILD_DATA;
my $build_data;
@BUILD_DATA = ReadFile "$BuildLog";
$build_data = join("\n", @BUILD_DATA);

my (@DEJAGNU_LOG, @DEJAGNU_SUM, @DEJAGNULOG_FULL, @GCC_VERSION);
my ($dejagnutests_log ,$dejagnutests_sum, $dejagnulog_full) = "";
my ($gcc_version, $gcc_version_long) = "";

$gcc_version_long="";
if ($GCCPATH ne "") {
	$gcc_version_long = `$GCCPATH/gcc --version`;
} elsif ($ENV{"CC"}) {
	$gcc_version_long = `$ENV{"CC"} --version`;
} else {
	$gcc_version_long = `gcc --version`;
}
@GCC_VERSION = split '\n', $gcc_version_long;
$gcc_version = $GCC_VERSION[0];

$llvmgcc_version_long="";
if ($LLVMGCCPATH ne "") {
  $llvmgcc_version_long = `$LLVMGCCPATH/llvm-gcc -v 2>&1`;
} else {
  $llvmgcc_version_long = `llvm-gcc -v 2>&1`;
}
@LLVMGCC_VERSION = split '\n', $llvmgcc_version_long;
$llvmgcc_versionTarget = $LLVMGCC_VERSION[1];
$llvmgcc_versionTarget =~ /Target: (.+)/;
$targetTriple = $1;

if(!$BuildError){
  @DEJAGNU_LOG = ReadFile "$DejagnuLog";
  @DEJAGNU_SUM = ReadFile "$DejagnuSum";
  $dejagnutests_log = join("\n", @DEJAGNU_LOG);
  $dejagnutests_sum = join("\n", @DEJAGNU_SUM);

  @DEJAGNULOG_FULL = ReadFile "$DejagnuTestsLog";
  $dejagnulog_full = join("\n", @DEJAGNULOG_FULL);
}

##############################################################
#
# Send data via a post request
#
##############################################################

if ( $VERBOSE ) { print "SEND THE DATA VIA THE POST REQUEST\n"; }

my %hash_of_data = (
  'machine_data' => $machine_data,
  'build_data' => $build_data,
  'gcc_version' => $gcc_version,
  'nickname' => $nickname,
  'dejagnutime_wall' => $DejagnuWallTime,
  'dejagnutime_cpu' => $DejagnuTime,
  'cvscheckouttime_wall' => $CheckoutTime_Wall,
  'cvscheckouttime_cpu' => $CheckoutTime_CPU,
  'configtime_wall' => $ConfigWallTime,
  'configtime_cpu'=> $ConfigTime,
  'buildtime_wall' => $BuildWallTime,
  'buildtime_cpu' => $BuildTime,
  'warnings' => "",
  'cvsusercommitlist' => "",
  'cvsuserupdatelist' => "",
  'cvsaddedfiles' => "",
  'cvsmodifiedfiles' => "",
  'cvsremovedfiles' => "",
  'lines_of_code' => "",
  'cvs_file_count' => 0,
  'cvs_dir_count' => 0,
  'buildstatus' => $BuildStatus,
  'singlesource_programstable' => $SingleSourceProgramsTable,
  'multisource_programstable' => $MultiSourceProgramsTable,
  'externalsource_programstable' => $ExternalProgramsTable,
  'llcbeta_options' => $multisource_llcbeta_options,
  'warnings_removed' => "",
  'warnings_added' => "",
  'passing_tests' => $passes,
  'expfail_tests' => $xfails,
  'unexpfail_tests' => $fails,
  'all_tests' => $dejagnu_test_list,
  'new_tests' => "",
  'removed_tests' => "",
  'dejagnutests_results' => $DejagnuTestResults,
  'dejagnutests_log' => $dejagnulog_full,
  'starttime' => $starttime,
  'endtime' => $endtime,
  'o_file_sizes' => "",
  'a_file_sizes' => "",
  'target_triple' => $targetTriple
);

if ($SUBMIT || !($SUBMITAUX eq "")) {
  my $response = SendData $SUBMITSERVER,$SUBMITSCRIPT,\%hash_of_data;
  if( $VERBOSE) { print "============================\n$response"; }
} else {
  print "============================\n";
  foreach $x(keys %hash_of_data){
      print "$x  => $hash_of_data{$x}\n";
  }
}

##############################################################
#
# Remove the source tree...
#
##############################################################
system ( "$NICE rm -rf $BuildDir")
  if (!$NOCHECKOUT and !$NOREMOVE and !$NOREMOVEATEND);
system ( "$NICE rm -rf $WebDir")
  if (!$NOCHECKOUT and !$NOREMOVE and !$NOREMOVERESULTS);
