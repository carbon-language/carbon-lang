#!/usr/dcs/software/supported/bin/perl -w
#
# Program:  NightlyTest.pl
#
# Synopsis: Perform a series of tests which are designed to be run nightly.
#           This is used to keep track of the status of the LLVM tree, tracking
#           regressions and performance changes.  This generates one web page a
#           day which can be used to access this information.
#
# Syntax:   NightlyTest.pl <CVSRootDir> <BuildDir> <WebDir>
#
use POSIX qw(strftime);

# Command line argument settings...
my $NOCHECKOUT = 0;
my $NOREMOVE   = 0;
my $MAKEOPTS   = "";

# Parse arguments...
while (scalar(@ARGV) and ($_ = $ARGV[0], /^[-+]/)) {
  shift;
  last if /^--$/;  # Stop processing arguments on --

  # List command line options here...
  if (/^-nocheckout$/) { $NOCHECKOUT = 1; next; }
  if (/^-noremove$/)   { $NOREMOVE   = 1; next; }
  if (/^-parallel$/)   { $MAKEOPTS   = "-j2 -l3.0"; next; }

  print "Unknown option: $_ : ignoring!\n";
}

die "Must specify 0 or 4 options!" if (@ARGV != 0 and @ARGV != 4);

my $HOME = $ENV{HOME};
my $CVSRootDir = "/home/vadve/vadve/Research/DynOpt/CVSRepository";
my $BuildDir   = "$HOME/buildtest";
my $WebDir     = "$HOME/cvs/testresults-X86";

# FIXME: This should just be utils/...
my $Template   = "$HOME/llvm/utils/NightlyTestTemplate.html";

if (@ARGV == 3) {
  $CVSRootDir = $ARGV[0];
  $BuildDir   = $ARGV[1];
  $WebDir     = $ARGV[2];
}

# Calculate the date prefix...
@TIME = localtime;
my $DATE = sprintf "%4d-%02d-%02d", $TIME[5]+1900, $TIME[4]+1, $TIME[3];
my $DateString = strftime "%B %d, %Y", localtime;

my $Prefix = "$WebDir/$DATE";

if (0) {
  print "CVS Root = $CVSRootDir\n";
  print "BuildDir = $BuildDir\n";
  print "WebDir   = $WebDir\n";
  print "Prefix   = $Prefix\n";
}

# Create the CVS repository directory
if (!$NOCHECKOUT) {
  mkdir $BuildDir or die "Could not create CVS checkout directory!";
}
chdir $BuildDir or die "Could not change to CVS checkout directory!";

# Check out the llvm tree, saving CVS messages to the cvs log...
system "(time -p cvs -d $CVSRootDir co llvm) > $Prefix-CVS-Log.txt 2>&1"
  if (!$NOCHECKOUT);

chdir "llvm" or die "Could not change into llvm directory!";

# Read in the HTML template file...
undef $/;
open (TEMPLATEFILE, $Template) or die "Could not open file 'llvm/$Template'!";
my $TemplateContents = <TEMPLATEFILE>;
close(TEMPLATEFILE);

sub GetRegex {
  $_[1] =~ /$_[0]/;
  return $1;
}

# Get some static statistics about the current state of CVS
my $CVSCheckoutTime = GetRegex "([0-9.]+)", `grep '^real' $Prefix-CVS-Log.txt`;
my $NumFilesInCVS = `grep ^U $Prefix-CVS-Log.txt | wc -l` + 0;
my $NumDirsInCVS  = `grep '^cvs checkout' $Prefix-CVS-Log.txt | wc -l` + 0;
$LOC = GetRegex "([0-9]+) +total", `wc -l \`utils/getsrcs.sh\` | grep total`;

# Build the entire tree, saving build messages to the build log
if (!$NOCHECKOUT) {
  # Change the Makefile.config to build into the local directory...
  rename "Makefile.config", "Makefile.config.orig";
  system "sed '/^LLVM_OBJ_DIR/d' < Makefile.config.orig > Makefile.config";
  system "echo >> Makefile.config";
  system "echo 'LLVM_OBJ_DIR := .' >> Makefile.config";

  # Change the Makefile.config to not strip executables...
  system "echo 'KEEP_SYMBOLS := 1' >> Makefile.config";

  # Build the entire tree, capturing the output into $Prefix-Build-Log.txt
  system "(time -p gmake $MAKEOPTS) > $Prefix-Build-Log.txt 2>&1";
}

# Get some statistics about the build...
my @Linked = split '\n', `grep Linking $Prefix-Build-Log.txt`;
my $NumExecutables = scalar(grep(/executable/, @Linked));
my $NumLibraries   = scalar(grep(!/executable/, @Linked));
my $NumObjects     = `grep '^Compiling' $Prefix-Build-Log.txt | wc -l` + 0;
my $BuildTime = GetRegex "([0-9.]+)", `grep '^real' $Prefix-Build-Log.txt`;


sub AddPreTag {  # Add pre tags around nonempty list, or convert to "none"
  $_ = shift;
  if (length) { return "<pre>  $_</pre>"; } else { "<b>none</b><br>"; }
}

# Get warnings from the build
my @Warn = split "\n", `grep -E 'warning:|Entering dir' $Prefix-Build-Log.txt`;
my @Warnings;
my $CurDir = "";

foreach $Warning (@Warn) {
  if ($Warning =~ m/Entering directory \`([^\`]+)\'/) {
    $CurDir = $1;                 # Keep track of directory warning is in...
    if ($CurDir =~ m|$BuildDir/llvm/(.*)|) { # Remove buildir prefix if included
      $CurDir = $1;
    }
  } else {
    push @Warnings, "$CurDir/$Warning";     # Add directory to warning...
  }
}
my $WarningsList = AddPreTag join "\n  ", @Warnings;


# Get some statistics about CVS commits over the current day...
@CVSHistory = split "\n", `cvs history -D '1 day ago' -a -xAMROCGUW`;
#print join "\n", @CVSHistory; print "\n";

# Extract some information from the CVS history... use a hash so no duplicate
# stuff is stored.
my (%AddedFiles, %ModifiedFiles, %RemovedFiles,
    %UsersCommitted, %UsersUpdated);

# Loop over every record from the CVS history, filling in the hashes.
foreach $File (@CVSHistory) {
  my ($Type, $Date, $UID, $Rev, $Filename);
  if ($File =~ /([AMR]) ([-:0-9 ]+\+[0-9]+) ([^ ]+) ([0-9.]+) +([^ ]+) +([^ ]+)/) {
    ($Type, $Date, $UID, $Rev, $Filename) = ($1, $2, $3, $4, "$6/$5");
  } elsif ($File =~ /([OCGUW]) ([-:0-9 ]+\+[0-9]+) ([^ ]+)/) {
    ($Type, $Date, $UID, $Rev, $Filename) = ($1, $2, $3, "", "");
  }
  #print "Ty = $Type Date = '$Date' UID=$UID Rev=$Rev File = '$Filename'\n";

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

my $UserCommitList = join "\n  ", keys %UsersCommitted;
my $UserUpdateList = join "\n  ", keys %UsersUpdated;
my $AddedFilesList = AddPreTag join "\n  ", keys %AddedFiles;
my $ModifiedFilesList = AddPreTag join "\n  ", keys %ModifiedFiles;
my $RemovedFilesList = AddPreTag join "\n  ", keys %RemovedFiles;

# Get a list of the previous days that we can link to...
system "rm -f $WebDir/$DATE.html";   # Don't relist self if regenerating...
opendir DH, $WebDir;
my @PrevDays =
  map {s/.html//; $_} reverse sort grep /[-0-9]+.html/, readdir DH;
closedir DH;

splice @PrevDays, 20;  # Trim down list to something reasonable...

my $PrevDaysList =     # Format list for sidebar
  join "\n  ", map { "<a href=\"$_.html\">$_</a><br>" } @PrevDays;

#
# Remove the cvs tree...
#
system "rm -rf $BuildDir" if (!$NOCHECKOUT and !$NOREMOVE);

# Print out information...
if (0) {
  print "DateString: $DateString\n";
  print "CVS Checkout: $CVSCheckoutTime seconds\n";
  print "Files/Dirs/LOC in CVS: $NumFilesInCVS/$NumDirsInCVS/$LOC\n";

  print "Build Time: $BuildTime seconds\n";
  print "Libraries/Executables/Objects built: $NumLibraries/$NumExecutables/$NumObjects\n";

  print "WARNINGS:\n  $WarningsList\n";


  print "Users committed: $UserCommitList\n";
  print "Added Files: \n  $AddedFilesList\n";
  print "Modified Files: \n  $ModifiedFilesList\n";
  print "Removed Files: \n  $RemovedFilesList\n";

  print "Previous Days =\n  $PrevDaysList\n";
}

# Output the file...
chdir $WebDir or die "Could not change into web directory!";
my $Output;
eval "\$Output = <<ENDOFFILE;$TemplateContents\nENDOFFILE\n";
open(OUTFILE, ">$DATE.html") or die "Cannot open output file!";
print OUTFILE $Output;
close(OUTFILE);

# Change the index.html symlink...
system "ln -sf $DATE.html index.html";

sub AddRecord {
  my ($Val, $Filename) = @_;
  my @Records;
  if (open FILE, $Filename) {
    @Records = grep !/$DATE/, split "\n", <FILE>;
    close FILE;
  }
  push @Records, "$DATE: $Val";
  open FILE, ">$Filename" or die "Couldn't open data file $Filename";
  print FILE (join "\n", @Records), "\n";
  close FILE;
  return @Records;
}

# Add information to the files which accumulate information for graphs...
AddRecord($LOC, "running_loc.txt");
AddRecord($BuildTime, "running_build_time.txt");
