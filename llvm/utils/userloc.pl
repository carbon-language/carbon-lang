#!/usr/bin/perl -w
#
# Program:  userloc.pl
#
# Synopsis: This program uses "cvs annotate" to get a summary of how many lines
#           of code the various developres are responsible for. It takes one
#           argument, the directory to process. If the argument is not specified
#           then the cwd is used. The directory must be an LLVM tree checked out
#           from cvs. 
#
# Syntax:   userloc.pl [-recurse|-tag=tag|-html... <directory>...
#
# Options:
#           -recurse
#               Recurse through sub directories. Without this, only the
#               specified directory is examined
#           -tag=tag
#               Use "tag" to select the revision (as per cvs -r option)
#           -filedetails
#               Report details about lines of code in each file for each user
#           -html
#               Generate HTML output instead of text output

die "Usage userloc.pl [-recurse|-tag=tag|-html] <directories>..." 
  if ($#ARGV < 0);

my $tag = "";
my $recurse = 0;
my $html = 0;
my $debug = 0;
my $filedetails = "";
while ( substr($ARGV[0],0,1) eq '-' )
{
  if ($ARGV[0] eq "-recurse") {
    $recurse = 1;
  } elsif ($ARGV[0] =~ /-tag=.*/) {
    $tag = $ARGV[0];
    $tag =~ s#-tag=(.*)#$1#;
  } elsif ($ARGV[0] =~ /-filedetails/) {
    $filedetails = 1;
  } elsif ($ARGV[0] eq "-html") {
    $html = 1;
  } elsif ($ARGV[0] eq "-debug") {
    $debug = 1;
  } else {
    die "Invalid option: $ARGV[0]";
  }
  shift;
}

die "Usage userloc.pl [-recurse|-tag=tag|-html] <directories>..." 
  if ($#ARGV < 0);

my %Stats;
my %FileStats;

sub ValidateFile
{
  my $f = $_[0];
  my $d = $_[1];

  if ( $d =~ ".*autoconf.*")
  {
    return 1 if ($f eq "configure.ac");
    return 1 if ($f eq "AutoRegen.sh");
    return 0;
  }
  return 0 if ( "$f" eq "configure");
  return 0 if ( "$f" eq 'PPCPerfectShuffle.h' );
  return 0 if ( $f =~ /.*\.cvs/);
  return 1;
}

sub GetCVSFiles
{
  my $d = $_[0];
  my $files ="";
  open STATUS, "cvs -nfz6 status $d -l 2>/dev/null |" 
    || die "Can't 'cvs status'";
  while ( defined($line = <STATUS>) )
  {
    if ( $line =~ /^File:.*/ )
    {
      chomp($line);
      $line =~ s#^File: ([A-Za-z0-9._-]*)[ \t]*Status:.*#$1#;
      $files = "$files $d/$line" if (ValidateFile($line,$d));
    }

  }
  return $files;
}

my $annotate = "cvs -z6 annotate -lf ";
if (length($tag) > 0)
{
  $annotate = $annotate . " -r" . $tag;
}

sub ScanDir
{
  my $Dir = $_[0];
  my $files = GetCVSFiles($Dir);

  open (DATA,"$annotate $files 2>&1 |")
    || die "Can't read cvs annotation data";

  my $curfile = "";
  while ( defined($line = <DATA>) )
  {
    chomp($line);
    if ($line =~ '^Annotations for.*') {
      $curfile = $line;
      $curfile =~ s#^Annotations for ([[:print:]]*)#$1#;
    } elsif ($line =~ /^[0-9.]*[ \t]*\([^)]*\):/) {
      $uname = $line;
      $uname =~ s#^[0-9.]*[ \t]*\(([a-zA-Z0-9_.-]*) [^)]*\):.*#$1#;
      $Stats{$uname}++;
      if ($filedetails) {
        $FileStats{$uname} = {} unless exists $FileStats{$uname};
        ${$FileStats{$uname}}{$curfile}++;
      }
    }
  }
  close DATA;
}

sub ValidateDirectory
{
  my $d = $_[0];
  return 0 if (! -d "$d" || ! -d "$d/CVS");
  return 0 if ($d =~ /.*CVS.*/);
  return 0 if ($d =~ /.*Debug.*/);
  return 0 if ($d =~ /.*Release.*/);
  return 0 if ($d =~ /.*Profile.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/html.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/man.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/ps.*/);
  return 0 if ($d =~ /.*docs\/CommandGuide\/man.*/);
  return 0 if ($d =~ /.*docs\/HistoricalNotes.*/);
  return 0 if ($d =~ /.*docs\/img.*/);
  return 0 if ($d =~ /.*bzip2.*/);
  return 1 if ($d =~ /.*projects\/Stacker.*/);
  return 1 if ($d =~ /.*projects\/sample.*/);
  return 0 if ($d =~ /.*projects\/llvm-.*/);
  return 0 if ($d =~ /.*win32.*/);
  return 0 if ($d =~ /.*\/.libs\/.*/);
  return 1;
}

sub printStats
{
  my $dir = $_[0];
  my $hash = $_[1];
  my $usr;
  my $total = 0;

  foreach $usr (keys %Stats) { $total += $Stats{$usr}; }

  if ($html) { 
    print "<table>";
    print " <tr><th style=\"text-align:right\">LOC</th>\n";
    print " <th style=\"text-align:right\">\%LOC</th>\n";
    print " <th style=\"text-align:left\">User</th>\n";
    print "</tr>\n";
  }

  foreach $usr ( sort keys %Stats )
  {
    my $v = $Stats{$usr};
    if (defined($v))
    {
      if ($html) {
        printf "<tr><td style=\"text-align:right\">%d</td><td style=\"text-align:right\">(%4.1f%%)</td><td style=\"text-align:left\">%s</td></tr>", $v, (100.0/$total)*$v,$usr;
      } else {
        printf "%8d  (%4.1f%%)  %s\n", $v, (100.0/$total)*$v, $usr;
      }
    }
  }
  print "</table>\n" if ($html);

  if ($filedetails) {
    foreach $user (sort keys %FileStats) {
      my $total = 0;
      foreach $file (sort keys %{$FileStats{$user}}) { 
        $total += ${$FileStats{$user}}{$file}
      }
      if ($html) {
        print "<table><tr><th style=\"text-align:left\" colspan=\"3\">$user</th></tr>\n";
      } else {
        print $user,":\n";
      }
      foreach $file (sort keys %{$FileStats{$user}}) {
        my $v = ${$FileStats{$user}}{$file};
        if ($html) { 
          printf "<tr><td style=\"text-align:right\">&nbsp;&nbsp;%d</td><td
          style=\"text-align:right\">&nbsp;%4.1f%%</td><td
          style=\"text-align:left\">%s</td></tr>",$v, (100.0/$total)*$v,$file;
        } else {
          printf "%8d  (%4.1f%%)  %s\n", $v, (100.0/$total)*$v, $file;
        }
      }
      if ($html) { print "</table>\n"; }
    }
  }
}

my @ALLDIRS = @ARGV;

if ($recurse)
{
  $Dirs = join(" ", @ARGV);
  $Dirs = `find $Dirs -type d \! -name CVS -print`;
  @ALLDIRS = split(' ',$Dirs);
}

if ($html)
{
print "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01//EN\" \"http://www.w3.org/TR/html4/strict.dtd\">\n";
print "<html>\n<head>\n";
print "  <title>LLVM LOC Based On CVS Annotation</title>\n";
print "  <link rel=\"stylesheet\" href=\"llvm.css\" type=\"text/css\"/>\n";
print "</head>\n";
print "<body><div class=\"doc_title\">LLVM LOC Based On CVS Annotation</div>\n";
print "<p>This document shows the total lines of code per user in each\n";
print "LLVM directory. Lines of code are attributed by the user that last\n";
print "committed the line. This does not necessarily reflect authorship.</p>\n";
}

my @ignored_dirs;

for $Dir (@ALLDIRS) 
{ 
  if ( ValidateDirectory($Dir) )
  {
    ScanDir($Dir); 
  }
  elsif ($html)
  {
    push @ignored_dirs, $Dir;
  }
}

printStats;

if ($html) 
{
  if (scalar @ignored_dirs > 0) {
    print "<p>The following directories were skipped:</p>\n";
    print "<ol>\n";
    foreach $index (0 .. $#ignored_dirs) {
      print " <li>", $ignored_dirs[$index], "</li>\n";
    }
    print "</ol>\n";
  }
  print "</body></html>\n";
}
