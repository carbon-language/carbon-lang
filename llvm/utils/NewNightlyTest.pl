#!/usr/bin/perl
use POSIX qw(strftime);
use File::Copy;
use Socket;

#
# Program:  NewNightlyTest.pl
#
# Synopsis: Perform a series of tests which are designed to be run nightly.
#           This is used to keep track of the status of the LLVM tree, tracking
#           regressions and performance changes. Submits this information 
#           to llvm.org where it is placed into the nightlytestresults database. 
#
# Modified heavily by Patrick Jenkins, July 2006
#
# Syntax:   NightlyTest.pl [OPTIONS] [CVSROOT BUILDDIR WEBDIR]
#   where
# OPTIONS may include one or more of the following:
#  -nocheckout      Do not create, checkout, update, or configure
#                   the source tree.
#  -noremove        Do not remove the BUILDDIR after it has been built.
#  -noremoveresults Do not remove the WEBDIR after it has been built.
#  -nobuild         Do not build llvm. If tests are enabled perform them
#                   on the llvm build specified in the build directory
#  -notest          Do not even attempt to run the test programs. Implies
#                   -norunningtests.
#  -norunningtests  Do not run the Olden benchmark suite with
#                   LARGE_PROBLEM_SIZE enabled.
#  -nodejagnu       Do not run feature or regression tests
#  -parallel        Run two parallel jobs with GNU Make.
#  -release         Build an LLVM Release version
#  -enable-llcbeta  Enable testing of beta features in llc.
#  -disable-llc     Disable LLC tests in the nightly tester.
#  -disable-jit     Disable JIT tests in the nightly tester.
#  -disable-cbe     Disable C backend tests in the nightly tester.
#  -verbose         Turn on some debug output
#  -debug           Print information useful only to maintainers of this script.
#  -nice            Checkout/Configure/Build with "nice" to reduce impact
#                   on busy servers.
#  -f2c             Next argument specifies path to F2C utility
#  -nickname        The next argument specifieds the nickname this script
#                   will submit to the nightlytest results repository.
#  -gccpath         Path to gcc/g++ used to build LLVM
#  -cvstag          Check out a specific CVS tag to build LLVM (useful for
#                   testing release branches)
#  -target          Specify the target triplet
#  -cflags          Next argument specifies that C compilation options that
#                   override the default.
#  -cxxflags        Next argument specifies that C++ compilation options that
#                   override the default.
#  -ldflags         Next argument specifies that linker options that override
#                   the default.
#  -compileflags    Next argument specifies extra options passed to make when
#                   building LLVM.
#  -use-gmake    		Use gmake instead of the default make command to build
#                   llvm and run tests.
#
#  ---------------- Options to configure llvm-test ----------------------------
#  -extraflags      Next argument specifies extra options that are passed to
#                   compile the tests.
#  -noexternals     Do not run the external tests (for cases where povray
#                   or SPEC are not installed)
#  -with-externals  Specify a directory where the external tests are located.
#
# CVSROOT is the CVS repository from which the tree will be checked out,
#  specified either in the full :method:user@host:/dir syntax, or
#  just /dir if using a local repo.
# BUILDDIR is the directory where sources for this test run will be checked out
#  AND objects for this test run will be built. This directory MUST NOT
#  exist before the script is run; it will be created by the cvs checkout
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
my $HOME = $ENV{'HOME'};
my $CVSRootDir = $ENV{'CVSROOT'};
   $CVSRootDir = "/home/vadve/shared/PublicCVS"
    unless $CVSRootDir;
my $BuildDir   = $ENV{'BUILDDIR'};
   $BuildDir   = "$HOME/buildtest"
    unless $BuildDir;
my $WebDir     = $ENV{'WEBDIR'};
   $WebDir     = "$HOME/cvs/testresults-X86"
    unless $WebDir;

##############################################################
#
# Calculate the date prefix...
#
##############################################################
@TIME = localtime;
my $DATE = sprintf "%4d-%02d-%02d", $TIME[5]+1900, $TIME[4]+1, $TIME[3];
my $DateString = strftime "%B %d, %Y", localtime;
my $TestStartTime = gmtime() . "GMT<br>" . localtime() . " (local)";

##############################################################
#
# Parse arguments...
#
##############################################################
$CONFIGUREARGS="";
$nickname="";
$NOTEST=0;
$NORUNNINGTESTS=0;
$MAKECMD="make";

while (scalar(@ARGV) and ($_ = $ARGV[0], /^[-+]/)) {
    shift;
    last if /^--$/;  # Stop processing arguments on --

  # List command line options here...
    if (/^-nocheckout$/)     { $NOCHECKOUT = 1; next; }
    if (/^-nocvsstats$/)     { $NOCVSSTATS = 1; next; }
    if (/^-noremove$/)       { $NOREMOVE = 1; next; }
    if (/^-noremoveresults$/) { $NOREMOVERESULTS = 1; next; }
    if (/^-notest$/)         { $NOTEST = 1; $NORUNNINGTESTS = 1; next; }
    if (/^-norunningtests$/) { $NORUNNINGTESTS = 1; next; }
    if (/^-parallel$/)       { $MAKEOPTS = "$MAKEOPTS -j2 -l3.0"; next; }
    if (/^-release$/)        { $MAKEOPTS = "$MAKEOPTS ENABLE_OPTIMIZED=1 ".
    						   						             "OPTIMIZE_OPTION=-O2"; 
    						               $BUILDTYPE="release"; next; }
    if (/^-enable-llcbeta$/) { $PROGTESTOPTS .= " ENABLE_LLCBETA=1"; next; }
    if (/^-disable-llc$/)    { $PROGTESTOPTS .= " DISABLE_LLC=1";
			       $CONFIGUREARGS .= " --disable-llc_diffs"; next; } 
    if (/^-disable-jit$/)    { $PROGTESTOPTS .= " DISABLE_JIT=1";
			       $CONFIGUREARGS .= " --disable-jit"; next; }
    if (/^-verbose$/)        { $VERBOSE = 1; next; }
    if (/^-debug$/)          { $DEBUG = 1; next; }
    if (/^-nice$/)           { $NICE = "nice "; next; }
    if (/^-f2c$/)            {
	$CONFIGUREARGS .= " --with-f2c=$ARGV[0]"; shift; next;
    }
    if (/^-with-externals/)  {
	$CONFIGUREARGS .= "--with-externals=$ARGV[0]"; shift; next;
    }
    if (/^-nickname$/)   	{ $nickname = "$ARGV[0]"; shift; next; }
    if (/^-gccpath/)         { $CONFIGUREARGS .= 
    													   " CC=$ARGV[0]/gcc CXX=$ARGV[0]/g++"; 
                               $GCCPATH=$ARGV[0]; 
                               shift;  
                               next;}
    else{ $GCCPATH=""; }
    if (/^-cvstag/)          { $CVSCOOPT .= " -r $ARGV[0]"; shift; next; } 
    else{ $CVSCOOPT="";}
    if (/^-target/)          {
	$CONFIGUREARGS .= " --target=$ARGV[0]"; shift; next;
    }
    if (/^-cflags/)          {
	$MAKEOPTS = "$MAKEOPTS C.Flags=\'$ARGV[0]\'"; shift; next;
    }
    if (/^-cxxflags/)        {
	$MAKEOPTS = "$MAKEOPTS CXX.Flags=\'$ARGV[0]\'"; shift; next;
    }
    if (/^-ldflags/)         {
	$MAKEOPTS = "$MAKEOPTS LD.Flags=\'$ARGV[0]\'"; shift; next;
    }
    if (/^-compileflags/)    {
	$MAKEOPTS = "$MAKEOPTS $ARGV[0]"; shift; next;
    }
    if (/^-use-gmake/)    {
			$MAKECMD = "gmake"; shift; next;
    }
    if (/^-extraflags/)      {
	$PROGTESTOPTS .= " EXTRA_FLAGS=\'$ARGV[0]\'"; shift; next;
    }
    if (/^-noexternals$/)    { $NOEXTERNALS = 1; next; }
    if (/^-nodejagnu$/)      { $NODEJAGNU = 1; next; }
    if (/^-nobuild$/)        { $NOBUILD = 1; next; }
    print "Unknown option: $_ : ignoring!\n";
}

if ($ENV{'LLVMGCCDIR'}) {
    $CONFIGUREARGS .= " --with-llvmgccdir=" . $ENV{'LLVMGCCDIR'};
}
if ($CONFIGUREARGS !~ /--disable-jit/) {
    $CONFIGUREARGS .= " --enable-jit";
}


if (@ARGV != 0 and @ARGV != 3){
	foreach $x (@ARGV){
		print "$x\n";
	}
	print "Must specify 0 or 3 options!";
}

if (@ARGV == 3) {
    $CVSRootDir = $ARGV[0];
    $BuildDir   = $ARGV[1];
    $WebDir     = $ARGV[2];
}

if($CVSRootDir eq "" or
   $BuildDir   eq "" or
   $WebDir     eq ""){
   die("please specify a cvs root directory, a build directory, and a ".
       "web directory");
 }
 
if($nickname eq ""){
	die ("Please invoke NewNightlyTest.pl with command line option \"-nickname <nickname>\"");
}

if($BUILDTYPE ne "release"){
	$BUILDTYPE = "debug";
}

##############################################################
#
#define the file names we'll use
#
##############################################################
my $Prefix = "$WebDir/$DATE";
my $BuildLog = "$Prefix-Build-Log.txt";
my $CVSLog = "$Prefix-CVS-Log.txt";
my $OldenTestsLog = "$Prefix-Olden-tests.txt";
my $SingleSourceLog = "$Prefix-SingleSource-ProgramTest.txt.gz";
my $MultiSourceLog = "$Prefix-MultiSource-ProgramTest.txt.gz";
my $ExternalLog = "$Prefix-External-ProgramTest.txt.gz";
my $DejagnuLog = "$Prefix-Dejagnu-testrun.log";
my $DejagnuSum = "$Prefix-Dejagnu-testrun.sum";
my $DejagnuTestsLog = "$Prefix-DejagnuTests-Log.txt";
if (! -d $WebDir) {
    mkdir $WebDir, 0777;
    warn "$WebDir did not exist; creating it.\n";
}

if ($VERBOSE) {
    print "INITIALIZED\n";
    print "CVS Root = $CVSRootDir\n";
    print "BuildDir = $BuildDir\n";
    print "WebDir   = $WebDir\n";
    print "Prefix   = $Prefix\n";
    print "CVSLog   = $CVSLog\n";
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# DiffFiles - Diff the current version of the file against the last version of
# the file, reporting things added and removed.  This is used to report, for
# example, added and removed warnings.  This returns a pair (added, removed)
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub DiffFiles {
    my $Suffix = shift;
    my @Others = GetDir $Suffix;
    if (@Others == 0) {  # No other files?  We added all entries...
	return (`cat $WebDir/$DATE$Suffix`, "");
    }
  # Diff the files now...
    my @Diffs = split "\n", `diff $WebDir/$DATE$Suffix $WebDir/$Others[0]`;
    my $Added   = join "\n", grep /^</, @Diffs;
    my $Removed = join "\n", grep /^>/, @Diffs;
    $Added =~ s/^< //gm;
    $Removed =~ s/^> //gm;
    return ($Added, $Removed);
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub GetRegex {   # (Regex with ()'s, value)
    $_[1] =~ /$_[0]/m;
    if (defined($1)) {
	return $1;
    }
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
    if(!$result){
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub GetDejagnuTestResults { # (filename, log)
    my ($filename, $DejagnuLog) = @_;
    my @lines;
    my $firstline;
    $/ = "\n"; #Make sure we're going line at a time.

    print "DEJAGNU TEST RESULTS:\n";

    if (open SRCHFILE, $filename) {
    # Process test results
	my $first_list = 1;
	my $should_break = 1;
	my $nocopy = 0;
	my $readingsum = 0;
	while ( <SRCHFILE> ) {
	    if ( length($_) > 1 ) {
		chomp($_);
		if ( m/^XPASS:/ || m/^FAIL:/ ) {
		    $nocopy = 0;
		    if ( $first_list ) {
			push(@lines, "UNEXPECTED TEST RESULTS\n");
			$first_list = 0;
			$should_break = 1;
			push(@lines, "$_\n");
			print "  $_\n";
		    } else {
			push(@lines, "$_\n");
			print "  $_\n";
		    }
		} #elsif ( m/Summary/ ) {
		#    if ( $first_list ) {
		#	push(@lines, "PERFECT!");
		#	print "  PERFECT!\n";
		#    } else {
		#	push(@lines, "</li></ol>\n");
		#    }
		#    push(@lines, "STATISTICS\n");
		#    print "\nDEJAGNU STATISTICS:\n";
		#    $should_break = 0;
		#    $nocopy = 0;
		#    $readingsum = 1;
		#} 
		elsif ( $readingsum ) {
		    push(@lines,"$_\n");
		    print "  $_\n";
		}

	    }
	}
    }
    close SRCHFILE;

    my $content = join("", @lines);
    return $content;
}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This function acts as a mini web browswer submitting data
# to our central server via the post method
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sub SendData{
    $host = $_[0];
    $file = $_[1];
    $variables=$_[2];

    $port=80;
    $socketaddr= sockaddr_in $port, inet_aton $host or die "Bad hostname\n";
    socket SOCK, PF_INET, SOCK_STREAM, getprotobyname('tcp') or 
    			 die "Bad socket\n";
    connect SOCK, $socketaddr or die "Bad connection\n";
    select((select(SOCK), $| = 1)[0]);

    #creating content here
    my $content;
    foreach $key (keys (%$variables)){
        $value = $variables->{$key};
        $value =~ s/([^A-Za-z0-9])/sprintf("%%%02X", ord($1))/seg;
        $content .= "$key=$value&";
    }

    $length = length($content);

    my $send= "POST $file HTTP/1.0\n";
    $send.= "Content-Type: application/x-www-form-urlencoded\n";
    $send.= "Content-length: $length\n\n";
    $send.= "$content";

    print SOCK $send;
    my $result;
    while(<SOCK>){
        $result  .= $_;
    }
    close(SOCK);

    my $sentdata="";
    foreach $x (keys (%$variables)){
        $value = $variables->{$x};
        $sentdata.= "$x  => $value\n";
    }
    WriteFile "$Prefix-sentdata.txt", $sentdata;
    

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
# Create the CVS repository directory
#
##############################################################
if (!$NOCHECKOUT) {
    if (-d $BuildDir) {
	if (!$NOREMOVE) {
		if ( $VERBOSE ){
			print "Build directory exists! Removing it\n";
		}
	    system "rm -rf $BuildDir";
	} else {
	    die "CVS checkout directory $BuildDir already exists!";
	}
    }
    mkdir $BuildDir or die "Could not create CVS checkout directory $BuildDir!";
}
ChangeDir( $BuildDir, "CVS checkout directory" );


##############################################################
#
# Check out the llvm tree, saving CVS messages to the cvs log...
#
##############################################################
my $CVSOPT = "";
# Use compression if going over ssh.
$CVSOPT = "-z3" if $CVSRootDir =~ /^:ext:/;
my $CVSCMD = "$NICE cvs $CVSOPT -d $CVSRootDir co $CVSCOOPT";
if (!$NOCHECKOUT) {
    if ( $VERBOSE ) 
    { 
	print "CHECKOUT STAGE:\n"; 
	print "( time -p $CVSCMD llvm; cd llvm/projects ; $CVSCMD llvm-test ) > $CVSLog 2>&1\n";
    }
    system "( time -p $CVSCMD llvm; cd llvm/projects ; " .
	"$CVSCMD llvm-test ) > $CVSLog 2>&1";
    ChangeDir( $BuildDir , "CVS Checkout directory") ;
}
ChangeDir( "llvm" , "llvm source directory") ;
if (!$NOCHECKOUT) {
    if ( $VERBOSE ) { print "UPDATE STAGE\n"; }
    system "$NICE cvs update -PdRA >> $CVSLog 2>&1" ;
}

##############################################################
#
# Get some static statistics about the current state of CVS
#
# This can probably be put on the server side
#
##############################################################
my $CVSCheckoutTime_Wall = GetRegex "([0-9.]+)", `grep '^real' $CVSLog`;
my $CVSCheckoutTime_User = GetRegex "([0-9.]+)", `grep '^user' $CVSLog`;
my $CVSCheckoutTime_Sys = GetRegex "([0-9.]+)", `grep '^sys' $CVSLog`;
my $CVSCheckoutTime_CPU = $CVSCheckoutTime_User + $CVSCheckoutTime_Sys;

my $NumFilesInCVS = `egrep '^U' $CVSLog | wc -l` + 0;
my $NumDirsInCVS  = `egrep '^cvs (checkout|server|update):' $CVSLog | wc -l` + 0;
my $LOC = `utils/countloc.sh`;

##############################################################
#
# Extract some information from the CVS history... use a hash so no duplicate
# stuff is stored. This gets the history from the previous days worth
# of cvs activit and parses it.
#
##############################################################

my (%AddedFiles, %ModifiedFiles, %RemovedFiles, %UsersCommitted, %UsersUpdated);

if(!$NOCVSSTATS){

if ($VERBOSE) { print "CVS HISTORY ANALYSIS STAGE\n"; }
@CVSHistory = split "\n", `cvs history -D '1 day ago' -a -xAMROCGUW`;
#print join "\n", @CVSHistory; print "\n";

my $DateRE = '[-/:0-9 ]+\+[0-9]+';

# Loop over every record from the CVS history, filling in the hashes.
foreach $File (@CVSHistory) {
    my ($Type, $Date, $UID, $Rev, $Filename);
    if ($File =~ /([AMRUGC]) ($DateRE) ([^ ]+) +([^ ]+) +([^ ]+) +([^ ]+)/) {
	($Type, $Date, $UID, $Rev, $Filename) = ($1, $2, $3, $4, "$6/$5");
    } elsif ($File =~ /([W]) ($DateRE) ([^ ]+)/) {
	($Type, $Date, $UID, $Rev, $Filename) = ($1, $2, $3, "", "");
    } elsif ($File =~ /([O]) ($DateRE) ([^ ]+) +([^ ]+)/) {
	($Type, $Date, $UID, $Rev, $Filename) = ($1, $2, $3, "", "$4/");
    } else {
	print "UNMATCHABLE: $File\n";
	next;
    }
    # print "$File\nTy = $Type Date = '$Date' UID=$UID Rev=$Rev File = '$Filename'\n";
    
    if ($Filename =~ /^llvm/) {
	if ($Type eq 'M') {        # Modified
	    $ModifiedFiles{$Filename} = 1;
	    $UsersCommitted{$UID} = 1;
	} elsif ($Type eq 'A') {   # Added
	    $AddedFiles{$Filename} = 1;
	    $UsersCommitted{$UID} = 1;
	} elsif ($Type eq 'R') {   # Removed
	    $RemovedFiles{$Filename} = 1;
	    $UsersCommitted{$UID} = 1;
	} else {
	    $UsersUpdated{$UID} = 1;
	}
    }
}

my $TestError = 1;

}#!NOCVSSTATS

my $CVSAddedFiles = join "\n", sort keys %AddedFiles;
my $CVSModifiedFiles = join "\n", sort keys %ModifiedFiles;
my $CVSRemovedFiles = join "\n", sort keys %RemovedFiles;
my $UserCommitList = join "\n", sort keys %UsersCommitted;
my $UserUpdateList = join "\n", sort keys %UsersUpdated;

##############################################################
#
# Build the entire tree, saving build messages to the build log
#
##############################################################
if (!$NOCHECKOUT && !$NOBUILD) {
    my $EXTRAFLAGS = "--enable-spec --with-objroot=.";
    if ( $VERBOSE ){
        print "CONFIGURE STAGE:\n";
        print "(time -p $NICE ./configure $CONFIGUREARGS $EXTRAFLAGS) > $BuildLog 2>&1\n";
    }
    system "(time -p $NICE ./configure $CONFIGUREARGS $EXTRAFLAGS) > $BuildLog 2>&1";
    if ( $VERBOSE ) 
    { 
			print "BUILD STAGE:\n";
			print "(time -p $NICE $MAKECMD $MAKEOPTS) >> $BuildLog 2>&1\n";
    }
    # Build the entire tree, capturing the output into $BuildLog
    system "(time -p $NICE $MAKECMD $MAKEOPTS) >> $BuildLog 2>&1";
}


##############################################################
#
# Get some statistics about the build...
#
##############################################################
#this can de done on server
#my @Linked = split '\n', `grep Linking $BuildLog`;
#my $NumExecutables = scalar(grep(/executable/, @Linked));
#my $NumLibraries   = scalar(grep(!/executable/, @Linked));
#my $NumObjects     = `grep ']\: Compiling ' $BuildLog | wc -l` + 0;


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
if($NOBUILD){
    $BuildStatus = "Skipped by user";
    $BuildError = 1;
}
elsif (`grep '^$MAKECMD\[^:]*: .*Error' $BuildLog | wc -l` + 0 ||
    `grep '^$MAKECMD: \*\*\*.*Stop.' $BuildLog | wc -l`+0) {
    $BuildStatus = "Error: compilation aborted";
    $BuildError = 1;
    print  "\n***ERROR BUILDING TREE\n\n";
}
if ($BuildError) { $NODEJAGNU=1; }

my $a_file_sizes="";
my $o_file_sizes="";
if(!$BuildError){
	if ( $VERBOSE ){
        print "Organizing size of .o and .a files\n";
    }
	ChangeDir( "$BuildDir/llvm", "Build Directory" );
	$afiles = `find . -iname '*.a' -ls`;
	$ofiles = `find . -iname '*.o' -ls`;
	@AFILES = split "\n", $afiles;
	$a_file_sizes="";
	foreach $x (@AFILES){
	  $x =~ m/.+\s+.+\s+.+\s+.+\s+.+\s+.+\s+(.+)\s+.+\s+.+\s+.+\s+(.+)/;
	  $a_file_sizes.="$1 $2 $BUILDTYPE\n";
	}	
	@OFILES = split "\n", $ofiles;
	$o_file_sizes="";
	foreach $x (@OFILES){
	  $x =~ m/.+\s+.+\s+.+\s+.+\s+.+\s+.+\s+(.+)\s+.+\s+.+\s+.+\s+(.+)/;
	  $o_file_sizes.="$1 $2 $BUILDTYPE\n";
	}
}
else{
	$a_file_sizes="No data due to a bad build.";
	$o_file_sizes="No data due to a bad build.";
}


##############################################################
#
# Running dejagnu tests
#
##############################################################
my $DejangnuTestResults; # String containing the results of the dejagnu
my $dejagnu_output = "$DejagnuTestsLog";
if(!$NODEJAGNU) {
    if($VERBOSE) 
    { 
			print "DEJAGNU FEATURE/REGRESSION TEST STAGE:\n"; 
			print "(time -p $MAKECMD $MAKEOPTS check) > $dejagnu_output 2>&1\n";
    }

    #Run the feature and regression tests, results are put into testrun.sum
    #Full log in testrun.log
    system "(time -p $MAKECMD $MAKEOPTS check) > $dejagnu_output 2>&1";
    
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
$DejagnuTestResults = "Dejagnu skipped by user choice." unless $DejagnuTestResults;
$DejagnuTime     = "0.0" unless $DejagnuTime;
$DejagnuWallTime = "0.0" unless $DejagnuWallTime;

if ($DEBUG) {
    print $DejagnuTestResults;
}    

##############################################################
#
# Get warnings from the build
#
##############################################################
if(!$NODEJAGNU){

if ( $VERBOSE ) { print "BUILD INFORMATION COLLECTION STAGE\n"; }
my @Warn = split "\n", `egrep 'warning:|Entering dir' $BuildLog`;
my @Warnings;
my $CurDir = "";

foreach $Warning (@Warn) {
    if ($Warning =~ m/Entering directory \`([^\`]+)\'/) {
	$CurDir = $1;                 # Keep track of directory warning is in...
	if ($CurDir =~ m#$BuildDir/llvm/(.*)#) { # Remove buildir prefix if included
	    $CurDir = $1;
	}
    } else {
	push @Warnings, "$CurDir/$Warning";     # Add directory to warning...
    }
}
my $WarningsFile =  join "\n", @Warnings;
$WarningsFile =~ s/:[0-9]+:/::/g;

# Emit the warnings file, so we can diff...
WriteFile "$WebDir/$DATE-Warnings.txt", $WarningsFile . "\n";
my ($WarningsAdded, $WarningsRemoved) = DiffFiles "-Warnings.txt";

# Output something to stdout if something has changed
#print "ADDED   WARNINGS:\n$WarningsAdded\n\n" if (length $WarningsAdded);
#print "REMOVED WARNINGS:\n$WarningsRemoved\n\n" if (length $WarningsRemoved);

#my @TmpWarningsAdded = split "\n", $WarningsAdded; ~PJ on upgrade
#my @TmpWarningsRemoved = split "\n", $WarningsRemoved; ~PJ on upgrade

} #endif !NODEGAGNU

##############################################################
#
# If we built the tree successfully, run the nightly programs tests...
#
# A set of tests to run is passed in (i.e. "SingleSource" "MultiSource" "External")
#
##############################################################
sub TestDirectory {
	my $SubDir = shift;
	
	ChangeDir( "$BuildDir/llvm/projects/llvm-test/$SubDir", "Programs Test Subdirectory" ) || return ("", "");
	
	my $ProgramTestLog = "$Prefix-$SubDir-ProgramTest.txt";
	
	# Run the programs tests... creating a report.nightly.csv file
	if (!$NOTEST) {
		print "$MAKECMD -k $MAKEOPTS $PROGTESTOPTS report.nightly.csv ".
          "TEST=nightly > $ProgramTestLog 2>&1\n";
		system "$MAKECMD -k $MAKEOPTS $PROGTESTOPTS report.nightly.csv ".
           "TEST=nightly > $ProgramTestLog 2>&1";
	  $llcbeta_options=`$MAKECMD print-llcbeta-option`;
	} 
    
  my $ProgramsTable;
  if (`grep '^$MAKECMD\[^:]: .*Error' $ProgramTestLog | wc -l` + 0){
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
           "| sort > $Prefix-multisourceprogramstable.txt";
  }
  $ProgramsTable = ReadFile "report.nightly.csv";

  ChangeDir( "../../..", "Programs Test Parent Directory" );
  return ($ProgramsTable, $llcbeta_options);
}

if (!$BuildError) {
	if ( $VERBOSE ) {
    print "SingleSource TEST STAGE\n";
	}
	($SingleSourceProgramsTable, $llcbeta_options) = TestDirectory("SingleSource");
	if ( $VERBOSE ) {
    print "SingleSource returned $SingleSourceProgramsTable\n";
	}
	WriteFile "$Prefix-singlesourceprogramstable.txt", $SingleSourceProgramsTable;
	if ( $VERBOSE ) {
	  print "MultiSource TEST STAGE\n";
	}
	($MultiSourceProgramsTable, $llcbeta_options) = TestDirectory("MultiSource");
	WriteFile "$Prefix-multisourceprogramstable.txt", $MultiSourceProgramsTable;
	if ( $VERBOSE ) {
	  print "MultiSource returned $MultiSourceProgramsTable\n";
	}
	if ( ! $NOEXTERNALS ) {
	  if ( $VERBOSE ) {
		  print "External TEST STAGE\n";
	  }
	  ($ExternalProgramsTable, $llcbeta_options) = TestDirectory("External");
	  WriteFile "$Prefix-externalprogramstable.txt", $ExternalProgramsTable;
	  system "cat $Prefix-singlesourceprogramstable.txt $Prefix-multisourceprogramstable.txt ".
		       " $Prefix-externalprogramstable.txt | sort > $Prefix-Tests.txt";
	} else {
	  $ExternalProgramsTable = "External TEST STAGE SKIPPED\n";
	  if ( $VERBOSE ) {
		  print "External TEST STAGE SKIPPED\n";
	  }
	  system "cat $Prefix-singlesourceprogramstable.txt $Prefix-multisourceprogramstable.txt ".
		       " | sort > $Prefix-Tests.txt";
	}
	WriteFile "$Prefix-externalprogramstable.txt", $ExternalProgramsTable;
}

##############################################################
#
# 
# gathering tests added removed broken information here
#
#
##############################################################
my $dejagnu = ReadFile $DejagnuSum;
my @DEJAGNU = split "\n", $dejagnu;

my $passes="",
my $fails="";
my $xfails="";

if(!$NODEJAGNU) {
	for($x=0; $x<@DEJAGNU; $x++){
		if($DEJAGNU[$x] =~ m/^PASS:/){
			$passes.="$DEJAGNU[$x]\n";
		}
		elsif($DEJAGNU[$x] =~ m/^FAIL:/){
			$fails.="$DEJAGNU[$x]\n";
		}
		elsif($DEJAGNU[$x] =~ m/^XFAIL:/){
			$xfails.="$DEJAGNU[$x]\n";
		}
	}
}

# my ($TestsAdded, $TestsRemoved, $TestsFixed, $TestsBroken) = ("","","","");
# 
# if ($TestError) {
#     $TestsAdded   = "<b>error testing</b><br>";
#     $TestsRemoved = "<b>error testing</b><br>";
#     $TestsFixed   = "<b>error testing</b><br>";
#     $TestsBroken  = "<b>error testing</b><br>";
# } else {
#     my ($RTestsAdded, $RTestsRemoved) = DiffFiles "-Tests.txt";
# 
#     my @RawTestsAddedArray = split '\n', $RTestsAdded;
#     my @RawTestsRemovedArray = split '\n', $RTestsRemoved;
# 
#     my %OldTests = map {GetRegex('TEST-....: (.+)', $_)=>$_}
#     @RawTestsRemovedArray;
#     my %NewTests = map {GetRegex('TEST-....: (.+)', $_)=>$_}
#     @RawTestsAddedArray;
# 
#     foreach $Test (keys %NewTests) {
# 			if (!exists $OldTests{$Test}) {  # TestAdded if in New but not old
# 	    	$TestsAdded = "$TestsAdded$Test\n";
# 			} else {
# 	    if ($OldTests{$Test} =~ /TEST-PASS/) {  # Was the old one a pass?
# 				$TestsBroken = "$TestsBroken$Test\n";  # New one must be a failure
# 	    } else {
# 				$TestsFixed = "$TestsFixed$Test\n";    # No, new one is a pass.
# 	    }
# 		}
# 	}
# 	foreach $Test (keys %OldTests) {  # TestRemoved if in Old but not New
# 		$TestsRemoved = "$TestsRemoved$Test\n" if (!exists $NewTests{$Test});
# 	}
# 
#     #print "\nTESTS ADDED:  \n\n$TestsAdded\n\n"   if (length $TestsAdded);
#     #print "\nTESTS REMOVED:\n\n$TestsRemoved\n\n" if (length $TestsRemoved);
#     #print "\nTESTS FIXED:  \n\n$TestsFixed\n\n"   if (length $TestsFixed);
#     #print "\nTESTS BROKEN: \n\n$TestsBroken\n\n"  if (length $TestsBroken);
# 
#     #$TestsAdded   = AddPreTag $TestsAdded;
#     #$TestsRemoved = AddPreTag $TestsRemoved;
#     #$TestsFixed   = AddPreTag $TestsFixed;
#     #$TestsBroken  = AddPreTag $TestsBroken;
# }

##############################################################
#
# If we built the tree successfully, runs of the Olden suite with
# LARGE_PROBLEM_SIZE on so that we can get some "running" statistics.
#
##############################################################
if (!$BuildError) {
    if ( $VERBOSE ) { print "OLDEN TEST SUITE STAGE\n"; }
    my ($NATTime, $CBETime, $LLCTime, $JITTime, $OptTime, $BytecodeSize,
	$MachCodeSize) = ("","","","","","","");
    if (!$NORUNNINGTESTS) {
	ChangeDir( "$BuildDir/llvm/projects/llvm-test/MultiSource/Benchmarks/Olden",
		   "Olden Test Directory");

	# Clean out previous results...
	system "$NICE $MAKECMD $MAKEOPTS clean > /dev/null 2>&1";
	
	# Run the nightly test in this directory, with LARGE_PROBLEM_SIZE and
	# GET_STABLE_NUMBERS enabled!
	if( $VERBOSE ) { print "$MAKECMD -k $MAKEOPTS $PROGTESTOPTS report.nightly.csv.out TEST=nightly " .
			     " LARGE_PROBLEM_SIZE=1 GET_STABLE_NUMBERS=1 > /dev/null 2>&1\n"; }
	system "$MAKECMD -k $MAKEOPTS $PROGTESTOPTS report.nightly.csv.out TEST=nightly " .
	    " LARGE_PROBLEM_SIZE=1 GET_STABLE_NUMBERS=1 > /dev/null 2>&1";
	system "cp report.nightly.csv $OldenTestsLog";
    } #else {
	#system "gunzip ${OldenTestsLog}.gz";
    #}
    
    # Now we know we have $OldenTestsLog as the raw output file.  Split
    # it up into records and read the useful information.
    #my @Records = split />>> ========= /, ReadFile "$OldenTestsLog";
    #shift @Records;  # Delete the first (garbage) record
    
    # Loop over all of the records, summarizing them into rows for the running
    # totals file.
    #my $WallTimeRE = "Time: ([0-9.]+) seconds \\([0-9.]+ wall clock";
    #foreach $Rec (@Records) {
	#my $rNATTime = GetRegex 'TEST-RESULT-nat-time: program\s*([.0-9m]+)', $Rec;
	#my $rCBETime = GetRegex 'TEST-RESULT-cbe-time: program\s*([.0-9m]+)', $Rec;
	#my $rLLCTime = GetRegex 'TEST-RESULT-llc-time: program\s*([.0-9m]+)', $Rec;
	#my $rJITTime = GetRegex 'TEST-RESULT-jit-time: program\s*([.0-9m]+)', $Rec;
	#my $rOptTime = GetRegex "TEST-RESULT-compile: .*$WallTimeRE", $Rec;
	#my $rBytecodeSize = GetRegex 'TEST-RESULT-compile: *([0-9]+)', $Rec;
	
	#$NATTime .= " " . FormatTime($rNATTime);
	#$CBETime .= " " . FormatTime($rCBETime);
	#$LLCTime .= " " . FormatTime($rLLCTime);
	#$JITTime .= " " . FormatTime($rJITTime);
	#$OptTime .= " $rOptTime";
	#$BytecodeSize .= " $rBytecodeSize";
    #}
    #
    # Now that we have all of the numbers we want, add them to the running totals
    # files.
    #AddRecord($NATTime, "running_Olden_nat_time.txt", $WebDir);
    #AddRecord($CBETime, "running_Olden_cbe_time.txt", $WebDir);
    #AddRecord($LLCTime, "running_Olden_llc_time.txt", $WebDir);
    #AddRecord($JITTime, "running_Olden_jit_time.txt", $WebDir);
    #AddRecord($OptTime, "running_Olden_opt_time.txt", $WebDir);
    #AddRecord($BytecodeSize, "running_Olden_bytecode.txt", $WebDir);
}

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
@CVS_DATA = ReadFile "$CVSLog";
$cvs_data = join("\n", @CVS_DATA);

my @BUILD_DATA;
my $build_data;
@BUILD_DATA = ReadFile "$BuildLog";
$build_data = join("\n", @BUILD_DATA);

my @DEJAGNU_LOG;
my @DEJAGNU_SUM;
my $dejagnutests_log;
my $dejagnutests_sum;
@DEJAGNU_LOG = ReadFile "$DejagnuLog";
@DEJAGNU_SUM = ReadFile "$DejagnuSum";
$dejagnutests_log = join("\n", @DEJAGNU_LOG);
$dejagnutests_sum = join("\n", @DEJAGNU_SUM);

my @DEJAGNULOG_FULL;
my $dejagnulog_full;
@DEJAGNULOG_FULL = ReadFile "$DejagnuTestsLog";
$dejagnulog_full = join("\n", @DEJAGNULOG_FULL);

my $gcc_version_long="";
if($GCCPATH ne ""){
  $gcc_version_long = `$GCCPATH/gcc --version`;
  print "$GCCPATH/gcc --version\n";
}
else{
  $gcc_version_long = `gcc --version`;
}
@GCC_VERSION = split '\n', $gcc_version_long;
my $gcc_version = $GCC_VERSION[0];

$all_tests = ReadFile, "$Prefix-Tests.txt";

##############################################################
#
# Send data via a post request
#
##############################################################

if ( $VERBOSE ) { print "SEND THE DATA VIA THE POST REQUEST\n"; }


my $host = "llvm.org";
my $file = "/nightlytest/NightlyTestAccept.cgi";
my %hash_of_data = ('machine_data' => $machine_data,
	       						'build_data' => $build_data,
               			'gcc_version' => $gcc_version,
						        'nickname' => $nickname,
	       						'dejagnutime_wall' => $DejagnuWallTime,
										'dejagnutime_cpu' => $DejagnuTime,
										'cvscheckouttime_wall' => $CVSCheckoutTime_Wall,
										'cvscheckouttime_cpu' => $CVSCheckoutTime_CPU,
										'configtime_wall' => $ConfigWallTime,
										'configtime_cpu'=> $ConfigTime,
										'buildtime_wall' => $BuildWallTime,
										'buildtime_cpu' => $BuildTime,
										'warnings' => $WarningsFile,
										'cvsusercommitlist' => $UserCommitList,
										'cvsuserupdatelist' => $UserUpdateList,
										'cvsaddedfiles' => $CVSAddedFiles,
										'cvsmodifiedfiles' => $CVSModifiedFiles,
										'cvsremovedfiles' => $CVSRemovedFiles,
										'lines_of_code' => $LOC,
										'cvs_file_count' => $NumFilesInCVS,
										'cvs_dir_count' => $NumDirsInCVS,
										'buildstatus' => $BuildStatus,
										'singlesource_programstable' => $SingleSourceProgramsTable,
										'multisource_programstable' => $MultiSourceProgramsTable,
										'externalsource_programstable' => $ExternalProgramsTable,
										'llcbeta_options' => $multisource_llcbeta_options,
										'warnings_removed' => $WarningsRemoved,
										'warnings_added' => $WarningsAdded,
										'passing_tests' => $passes,
										'expfail_tests' => $xfails,
										'unexpfail_tests' => $fails,
										'all_tests' => $all_tests,
										'new_tests' => "",
										'removed_tests' => "",
										'dejagnutests_log' => $dejagnutests_log,
										'dejagnutests_sum' => $dejagnutests_sum,
										'starttime' => $starttime,
										'endtime' => $endtime,
										'o_file_sizes' => $o_file_sizes,
										'a_file_sizes' => $a_file_sizes);

$TESTING = 0;

if($TESTING){
    print "============================\n";
    foreach $x(keys %hash_of_data){
        print "$x  => $hash_of_data{$x}\n";
    }
}
else{
    my $response = SendData $host,$file,\%hash_of_data;
    print "============================\n$response";
}

##############################################################
#
# Remove the cvs tree...
#
##############################################################
system ( "$NICE rm -rf $BuildDir") if (!$NOCHECKOUT and !$NOREMOVE);
system ( "$NICE rm -rf $WebDir") if (!$NOCHECKOUT and !$NOREMOVE and !$NOREMOVERESULTS);


