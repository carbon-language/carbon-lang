#!/usr/bin/env perl

# runtest [options] FILENAME
#
# Read the file FILENAME. Each line contains a test.
# Convert template to test and crosstest.
# If possilble generate orphaned testversions, too.
# Use make to compile the test

################################################################################
# Global configuration options for the runtestscript itself:
################################################################################

# name of the global configuration file for the testsuite:
$config_file    = "ompts.conf";
$logfile        = "ompts.log"; # overwriteable by value in config file
$env_set_threads_command = 'OMP_NUM_THREADS=%n; export OMP_NUM_THREADS;';
$debug_mode     = 0;
################################################################################
# After this line the script part begins! Do not edit anithing below
################################################################################


# Namespaces:
use Getopt::Long;
#use Unix::PID;
use Data::Dumper;
use ompts_parserFunctions;

# Extracting given options
GetOptions("help",
      "listlanguages",
      "lang=s",
      "list",
      "testinfo=s",
      "numthreads=i",
      "test=s",
      "compile!",
      "run!",
      "orphan!",
      "resultfile=s"
      );

# Get global configuratino options from config file:
if(! -e $config_file){ error ("Could not find config file $config_file\n", 1);}
open (CONFIG, "<$config_file") or error ("Could not open config file $config_file\n", 2);
while (<CONFIG>) { $config .= $_; }
close (CONFIG);

($logfile) = get_tag_values ("logfile", $config);
($timeout) = get_tag_values ("singletesttimeout", $config);
($display_errors) = get_tag_values("displayerrors", $config);
($display_warnings) = get_tag_values ("displaywarnings", $config);
($numthreads) = get_tag_values ("numthreads", $config);
($env_set_threads_command) = get_tag_values("envsetthreadscommand",$config);
$env_set_threads_command =~ s/\%n/$numthreads/g;
@languages = get_tag_values ("language", $config);

if (!defined($opt_compile)) {$opt_compile = 1;}
if (!defined($opt_run))     {$opt_run = 1;}
if (!defined($opt_orphan)) {$opt_orphan = 1;}
if (!defined($opt_resultsfile)) {($opt_resultsfile) = get_tag_values("resultsfile", $config);}
if ( defined($opt_numthreads) && ($opt_numthreads > 0)) {$numthreads = $opt_numthreads;}
if ($debug_mode) {
print <<EOF;
Testsuite configuration:
Logfile = $logfile
Timeout = $timeout seconds
Language:  $opt_lang
Display errors:   $display_errors
Display warnings: $display_warnings
Resultsfile:      $opt_resultsfile
Numthreads: $numthreads
------------------------------
EOF
}

$num_construcs    = 0;
$num_tests        = 0;
$num_failed_tests = 0;
$num_successful_tests   = 0;
$num_verified_tests     = 0;
$num_failed_compilation = 0;

$num_normal_tests_failed = 0;
$num_normal_tests_compile_error = 0;
$num_normal_tests_timed_out = 0;
$num_normal_tests_successful = 0;
$num_normal_tests_verified = 0;

$num_orphaned_tests_failed = 0;
$num_orphaned_tests_compile_error = 0;
$num_orphaned_tests_timed_out = 0;
$num_orphaned_tests_successful = 0;
$num_orphaned_tests_verified = 0;

if ($opt_help)         { print_help_text ();   exit 0; }
if ($opt_listlanguages){ print_avail_langs (); exit 0; }
if ($opt_list)     { print_avail_tests ();   exit 0; }
if ($opt_testinfo) { print_testinfo ();      exit 0; }
if ($opt_test)     { write_result_file_head();
                     execute_single_test (); exit 0; }
if (-e $ARGV[0])   { write_result_file_head();
                     execute_testlist($ARGV[0]); print_results();
                     result_summary(); exit 0;}

################################################################################
# sub function definitions
################################################################################

# Function which prints the results file
sub print_results
{
    system("echo; cat $opt_resultsfile; echo;");
}

# Function which prints a summary of all test
sub result_summary
{
    my $num_directives = @test_results;

    print <<EOF;

Summary:
S Number of tested Open MP constructs: $num_constructs
S Number of used tests:                $num_tests
S Number of failed tests:              $num_failed_tests
S Number of successful tests:          $num_successful_tests
S + from this were verified:           $num_verified_tests

Normal tests:
N Number of failed tests:              $num_normal_tests_failed
N + from this fail compilation:        $num_normal_tests_compile_error
N + from this timed out                $num_normal_tests_timed_out
N Number of successful tests:          $num_normal_tests_successful
N + from this were verified:           $num_normal_tests_verified

Orphaned tests:
O Number of failed tests:              $num_orphaned_tests_failed
O + from this fail compilation:        $num_orphaned_tests_compile_error
O + from this timed out                $num_orphaned_tests_timed_out
O Number of successful tests:          $num_orphaned_tests_successful
O + from this were verified:           $num_orphaned_tests_verified
EOF

}

# Function that executest the tests specified in the given list
sub execute_testlist
{
    my ($filename) = @_;
    # opening testlist
    open(TESTS,$filename) or error ("Could not open  $filename\n", 1);
TEST: while (<TESTS>) {
        if (/^\s*#/) {next TEST;}
        if (/^\s*$/) {next TEST;}
        $opt_test = $_;
        chomp ($opt_test);
        execute_single_test ();
    }
# print Dumper(@test_results);
}

# Function that executes a system command but takes care of the global timeout
# If command did not finish inbetween returns '-' otherwise the exit status of
# the system command
sub timed_sys_command
{
    my ($command) = @_;
    my $exit_status = '-';

# set up the timeout for the command
    eval {
        local $SIG{ALRM} = sub {die "alarm\n"};
        alarm $timeout;
        log_message_add ("Starting command \"$command\"");
        $exit_status = system ($command);
        alarm 0;
    };
# check if command finished during the maximum execution time
    if ($@ eq "alarm\n") {
# test timed out
#		my $pid = Unix::PID->new();
#		$pid->get_pidof($command, 1);
#		$pid->kill();
        if ($debug_mode) {
	    log_message_add ("Command \"$command\" reached max execution time.\n");
        }
        return "TO";
    }
# test finished
    return $exit_status;
}

# Function that runs the tests given as a array containing the testnames
# Returns an array containing the percent values of the passed tests and the
# successful crosstests.
sub run_test
{
    my ($testname, $orphan) = @_;
    my $bin_name, $cbin_name;
    my $cmd, $exit_status, $failed;
    my $resulttest, $resultctest;

# path to test and crosstest either in normal or in orphaned version
    if ($orphan) {
        $bin_name  = "bin/$opt_lang/orph_test_$testname";
        $cbin_name = "bin/$opt_lang/orph_ctest_$testname";
    } else {
        $bin_name  = "bin/$opt_lang/test_$testname";
        $cbin_name = "bin/$opt_lang/ctest_$testname";
    }
# Check if executables exist
    if (! -e $bin_name) {
        test_error ("Could not find executable \"$bin_name\".");
        return ('test' => '-', 'crosstest' => '-');
    }
# run the test
    $cmd = "$env_set_threads_command ./$bin_name >$bin_name.out";
    print "Running test with $numthreads threads .";
    $exit_status = timed_sys_command ($cmd);
############################################################
# Check if test finished within max execution time
    if ($exit_status eq 'TO') {
        print ".... failed (timeout)\n";
        return ('test' => 'TO', 'crosstest' => '-')
    }
############################################################
# check if all tests were successful
    $failed = $exit_status >> 8;
    if ($failed < 0 or $failed > 100) { $failed = 100; }
    $resulttest = 100 - $failed;
    if ($resulttest eq 100) {
        print ".... success ...";
    } else {
        print ".... failed $failed\% of the tests\n";
        return ('test' => $resulttest, 'crosstest' => '-');
    }
############################################################

# do crosstest
# check if executable exist
    if (! -e $cbin_name) {
        test_error ("Could not find executable \"$cbin_name\".");
        print "... not verified (crosstest missing)\n";
        return ('test' => $resulttest, 'crosstest' => '-');
    }
# run crosstest
# Test was successful, so it makes sense to run the crosstest
    $cmd = "$env_set_threads_command ./$cbin_name > $cbin_name.out";
    $exit_status = timed_sys_command ($cmd);
############################################################
# Check if crosstest finished within max execution time
    if ($exit_status eq 'TO') {
        print "... not verified (timeout)\n";
        return ('test' => $result, 'crosstest' => 'TO');
    }
############################################################
# test if crosstests failed as expected
    $resultctest = $exit_status >> 8;
    if ($resultctest > 0) {
        print "... and verified with $resultctest\% certainty\n";
    } else {
        print "... but might be lucky\n";
    }
    return ('test' => $resulttest, 'crosstest' => $resultctest);
############################################################
}

# Function that generates the test binaries out of the sourcecode
sub compile_src
{
    my ($testname, $orphan) = @_;
    print "Compiling soures ............";
    if ($orphan) {
# Make orphaned tests
        $exec_name     = "bin/$opt_lang/orph_test_$testname";
        $crossexe_name = "bin/$opt_lang/orph_ctest_$testname";
        $resulttest  = system ("make $exec_name > $exec_name\_compile.log" );
        $resultctest = system ("make $crossexe_name > $crossexe_name\_compile.log" );
    } else {
# Make test
        $exec_name     = "bin/$opt_lang/test_$testname";
        $crossexe_name = "bin/$opt_lang/ctest_$testname";
        $resulttest  = system ("make $exec_name > $exec_name\_compile.log" );
        $resultctest = system ("make $crossexe_name > $crossexe_name\_compile.log" );
    }
    if ($resulttest) { test_error ("Compilation of the test failed."); }
    if ($resultctest){ test_error ("Compilation of the crosstest failed."); }

    if ($resulttest or $resultctest) {
        print ".... failed\n";
        return 0;
    } else {
        print ".... success\n";
        return 1;
    }
}

# Function which prepare the directory structure:
sub init_directory_structure
{
    my ($language) = @_;
    if (-e "bin" && -d "bin") { warning ("Old binary directory detected!");}
    else { system ("mkdir bin"); }
    if (-e "bin/$language" && -d "bin/$language") {
        warning ("Old binary directory for language $language found.");}
    else { system ("mkdir bin/$language"); }
}

# Function that generates the sourcecode for the given test
sub make_src
{
    my ($testname, $orphan) = @_;
    my $template_file;
    my $src_name;

    $template_file = "$dir/$testname.$extension";
    if (!-e $template_file) { test_error ("Could not find template for \"$testname\""); }

    print "Generating sources ..........";
    if ($orphan) {
# Make orphaned tests
        $src_name = "bin/$opt_lang/orph_test_$testname.$extension";
        $resulttest = system ("./$templateparsername --test --orphan $template_file $src_name");
        $src_name = "bin/$opt_lang/orph_ctest_$testname.$extension";
        $resultctest = system ("./$templateparsername --crosstest --orphan $template_file $src_name");
    } else {
# Make test
        $src_name = "bin/$opt_lang/test_$testname.$extension";
        $resulttest = system ("./$templateparsername --test --noorphan $template_file $src_name");
        $src_name = "bin/$opt_lang/ctest_$testname.$extension";
        $resultctest = system ("./$templateparsername --crosstest --noorphan $template_file $src_name");
    }
    if ($resulttest) { test_error ("Generation of sourcecode for the test failed."); }
    if ($resultctest){ test_error ("Generation of sourcecode for the crosstest failed."); }

    if ($resulttest or $resultctest) {
        print ".... failed\n";
       return 0;
    } else {
       print ".... success\n";
       return 1;
    }
}

# Function which checks if a given test is orphanable
sub test_is_orphanable
{
    my ($testname) = @_;
    my $src;
    my $file = "$dir/$testname.$extension";
    if(! -e $file){ test_error ("Could not find test file $file\n");}
    open (TEST, "<$file") or test_error ("Could not open test file $file\n");
    while (<TEST>) { $src .= $_; }
    close (TEST);
    return $src =~/ompts:orphan/;
}

sub write_result_file_head
{
    open (RESULTS, ">$opt_resultsfile") or error ("Could not open file '$opt_resultsfile' to write results.", 1);
    $resultline = sprintf "%-25s %-s\n", "#Tested Directive", "\tt\tct\tot\toct";
    print RESULTS $resultline;
}

# Function which adds a result to the list of results
sub add_result
{
    my ($testname, $result) = @_;
#	print Dumper(@{$result});

    $num_constructs++;

    open (RESULTS, ">>$opt_resultsfile") or error ("Could not open file '$opt_resultsfile' to write results.", 1);

    if (${$result}[0][0]) {
		$num_tests ++;}

	if ($opt_compile and ${$result}[0][1] eq 0) {
		${$result}[0][2]{test}      = 'ce';
		${$result}[0][2]{crosstest} = '-';
		$num_normal_tests_compile_error++;
	    $num_normal_tests_failed++;
	}

    if ($opt_run and ${$result}[0][2] and ${$result}[0][2]{test} ne 'ce') {
        if (${$result}[0][2]{test} == 100) {
            $num_normal_tests_successful++;
            if (${$result}[0][2]{crosstest} == 100){
				$num_normal_tests_verified++;}
		} elsif (${$result}[0][2]{test} eq 'TO'){
			$num_normal_tests_timed_out++;
			$num_normal_tests_failed++;
		} else {
			$num_normal_tests_failed++;
		}
    }
    $resultline = "${$result}[0][2]{test}\t${$result}[0][2]{crosstest}\t";

    if (${$result}[1][0]) {
		$num_tests ++;}
    else { $resultline .= "-\t-\n"; }

    if ($opt_compile and ${$result}[1][1] eq 0) {
		${$result}[1][2]{test}      = 'ce';
		${$result}[1][2]{crosstest} = '-';
		$num_orphaned_tests_compile_error++;
		$num_orphaned_tests_failed++;
	}

    if ($opt_run and ${$result}[1][2] and ${$result}[1][2]{test} ne 'ce') {
        if (${$result}[1][2]{test} == 100) {
            $num_orphaned_tests_successful++;
            if (${$result}[1][2]{crosstest} == 100){
				$num_orphaned_tests_verified++;}
		} elsif (${$result}[1][2]{test} eq 'TO'){
			$num_orphaned_tests_timed_out++;
			$num_orphaned_tests_failed++;
        } else {
			$num_orphaned_tests_failed++;
		}
    }
    $resultline .= "${$result}[1][2]{test}\t${$result}[1][2]{crosstest}\n";

    $num_failed_tests = $num_normal_tests_failed + $num_orphaned_tests_failed;
	$num_failed_compilation = $num_normal_tests_compile_error + $num_orphaned_tests_compile_error;
	$num_successful_tests = $num_normal_tests_successful + $num_orphaned_tests_successful;
	$num_verified_tests = $num_normal_tests_verified + $num_orphaned_tests_verified;

    $resultline2 = sprintf "%-25s %-s", "$testname", "\t$resultline";
    print RESULTS $resultline2;
}

# Function which executes a single test
sub execute_single_test
{
    my @result;
    init_language_settings ($opt_lang);
    init_directory_structure ($opt_lang);
    log_message_add ("Testing for \"$opt_test\":");
    print "Testing for \"$opt_test\":\n";
# tests in normal mode
    if ($opt_compile){ $result[0][0] = make_src ($opt_test, 0);
                       $result[0][1] = compile_src ($opt_test, 0);}
    if ($opt_run && $result[0][1] == 1) {
                       $result[0][2] = {run_test ($opt_test, 0)};}
# tests in orphaned mode
    if ($opt_orphan && test_is_orphanable($opt_test)){
        log_message_add ("Testing for \"$opt_test\" in orphaned mode:");
        print "+ orphaned mode:\n";
        if ($opt_compile) { $result[1][0] = make_src ($opt_test, 1);
                            $result[1][1] = compile_src ($opt_test, 1);}
        if ($opt_run && $result[1][1] == 1) {
                            $result[1][2] = {run_test ($opt_test, 1)};}
    }
    add_result($opt_test, \@result);
}

# Function that prints info about a given test
sub print_testinfo
{
    init_language_settings($opt_lang);
    my $doc = "";
    my $file = $dir."/".$opt_testinfo.".".$extension;
    if (! -e $file) {error ("Could not find template for test $opt_testinfo", 5);}
    open (TEST,"<$file") or error ("Could not open template file \"$file\" for test $opt_testinfo", 6);
    while (<TEST>) {$doc .= $_;}
    close (TEST);

    (my $omp_version) = get_tag_values ("ompts:ompversion", $doc);
    (my $dependences) = get_tag_values ("ompts:dependences", $doc);
    (my $description) = get_tag_values ("ompts:testdescription", $doc);
    my $orphanable = 'no';
    if ($doc =~ /ompts:orphan/) {$orphanable = 'yes';}
    print <<EOF;
Info for test $opt_testinfo:
Open MP standard: $omp_version
Orphaned mode: $orphanable
Dependencies:  $dependences
Description:   $description
EOF
}

# Function that initializes the settings for the given language
sub init_language_settings
{
    my ($language) = @_;
    foreach my $lang (@languages) {
        (my $name) = get_tag_values ("languagename", $lang);
        if ($name eq $language) {
            ($extension) = get_tag_values ("fileextension", $lang);
            ($dir)       = get_tag_values ("dir", $lang);
            ($templateparsername) = get_tag_values ("templateparsername", $lang);
            last;
        }
    }
    # Check if we found the specified language in the config file
    if (!$extension and !$dir) {
      error ("Language $language could not be found.\n", 3);
    }
}



# Function that prints all available tests for the given language
sub print_avail_tests
{
    init_language_settings($opt_lang);
    my @tests;
    opendir(DIR,"$dir") or error ("Could not open directory $dir", 4);
    while($_ = readdir(DIR)) { if (/\.$extension$/) {s/\.$extension//; push (@tests, $_);}}
    closedir(DIR);
    print "Found ".(@tests)." tests:\n". "-" x 30 . "\n";
    foreach (@tests) { print $_."\n";}
}

# Function that prints all available tests for the given language
sub print_avail_langs
{
    if (@languages > 0) {
        print "Available languages:\n";
        foreach (@languages) {
            (my $name) = get_tag_values ("languagename", $_);
            print "$name\n";
        }
    } else {
        print "No languages available\n";
    }
}

# Function that prints the error message
sub print_help_text
{
    print <<EOF;
runtest.pl [options] [FILE]

Executes the tests listed in FILE. FILE has to contain the names of the tests,
one test per line. Lines starting with '#' will be ignored.
A language has to be specified for all commands except --help and --listlanguages.

Options:
  --help            displays this help message
  --listlanguages   lists all available languages
  --lang=s          select language
  --list            list available tests for a language
  --testinfo=NAME   show info for test NAME
  --numthreads=NUM  set number of threads (overwrites config file settings)
  --test=NAME       execute single test NAME
  --nocompile       do not compile tests
  --norun           do not run tests
  --noorphan        switch of orphaned tests
  --resultfile=NAME use NAME as resultfile (overwrites config file settings)
EOF
}

# Function that writes an error message for a failed test / part of a test
sub test_error
{
   my ($message) = @_;
   log_message_add ("ERROR: $message");
   if ($display_errors eq 1) { print STDERR "ERROR: $message\n"; }
}

# Function that returns an warning message
sub warning {
  my ($message) = @_;
  if ($display_warnings eq 1) { print "Warniong: $message\n"; }
  log_message_add ("Warning: $message");
}

# Function that returns an error message and exits with the specified error code
sub error {
  my ($message, $error_code) = @_;
  if ($display_errors eq 1) { print STDERR "ERROR: $message\n"; }
  log_message_add ("ERROR: $message");
  exit ($error_code);
}

# Function which adds an new entry into the logfile together with a timestamp
sub log_message_add
{
    (my $message) = @_;
    ($sec,$min,$hour,$mday,$mon,$year,$wday,$ydat,$isdst) = localtime();
    if(length($hour) == 1) { $hour="0$hour"; }
    if(length($min) == 1)  { $min="0$min";   }
    if(length($sec) == 1)  { $sec="0$sec";   }
    $mon=$mon+1;
    $year=$year+1900;
    open (LOGFILE,">>$logfile") or die "ERROR: Could not create $logfile\n";
    print LOGFILE "$mday/$mon/$year $hour.$min.$sec: $message\n";
}
