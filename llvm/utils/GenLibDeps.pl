#!/usr/bin/perl -w
#
# Program:  GenLibDeps.pl
#
# Synopsis: Generate HTML output that shows the dependencies between a set of
#           libraries. The output of this script should periodically replace 
#           the similar content in the UsingLibraries.html document.
#
# Syntax:   GenLibDeps.pl [-flat] <directory_with_libraries_in_it>
#

# Parse arguments... 
my $FLAT = 0;
my $WHY = 0;
while (scalar(@ARGV) and ($_ = $ARGV[0], /^[-+]/)) {
  shift;
  last if /^--$/;  # Stop processing arguments on --

  # List command line options here...
  if (/^-flat$/)     { $FLAT = 1; next; }
  if (/^-why/)       { $WHY = 1; $FLAT = 1; next; }
  print "Unknown option: $_ : ignoring!\n";
}

# Give first option a name.
my $Directory = $ARGV[0];

# Find the "dot" program
my $DotPath="";
if (!$FLAT) {
  chomp($DotPath = `which dot`);
  die "Can't find 'dot'" if (! -x "$DotPath");
}

chomp(my $nmPath=`which nm`);
die "Can't find 'nm'" if (! -x "$nmPath");

# Open the directory and read its contents, sorting by name and differentiating
# by whether its a library (.a) or an object file (.o)
opendir DIR,$Directory;
my @files = readdir DIR;
closedir DIR;
@libs = grep(/libLLVM.*\.a$/,sort(@files));
@objs = grep(/LLVM.*\.o$/,sort(@files));

# Declare the hashes we will use to keep track of the library and object file
# symbol definitions.
my %libdefs;
my %objdefs;

# Gather definitions from the libraries
foreach $lib (@libs ) {
  open DEFS, 
    "$nmPath -g $Directory/$lib | grep ' [ABCDGRST] ' | sed -e 's/^[0-9A-Fa-f]* [ABCDGRST] //' | sort | uniq |";
  while (<DEFS>) {
    chomp($_);
    $libdefs{$_} = $lib;
  }
  close DEFS;
}

# Gather definitions from the object files.
foreach $obj (@objs ) {
  open DEFS, 
    "$nmPath -g $Directory/$obj | grep ' [ABCDGRST] ' | sed -e 's/^[0-9A-Fa-f]* [ABCDGRST] //' | sort | uniq |";
  while (<DEFS>) {
    chomp($_);
    $objdefs{$_} = $obj;
  }
  close DEFS;
}

# Generate one entry in the <dl> list. This generates the <dt> and <dd> elements
# for one library or object file. The <dt> provides the name of the library or
# object. The <dd> provides a list of the libraries/objects it depends on.
sub gen_one_entry {
  my $lib = $_[0];
  my $lib_ns = $lib;
  $lib_ns =~ s/(.*)\.[oa]/$1/;
  if ($FLAT) {
    print "$lib:";
    if ($WHY) { print "\n"; }
  } else {
    print "  <dt><b>$lib</b</dt><dd><ul>\n";
  }
  open UNDEFS, 
    "$nmPath -g -u $Directory/$lib | sed -e 's/^  *U //' | sort | uniq |";
  my %DepLibs;
  while (<UNDEFS>) {
    chomp;
    my $lib_printed = 0;
    if (defined($libdefs{$_}) && $libdefs{$_} ne $lib) {
      $DepLibs{$libdefs{$_}} = [] unless exists $DepLibs{$libdefs{$_}};
      push(@{$DepLibs{$libdefs{$_}}}, $_);
    } elsif (defined($objdefs{$_}) && $objdefs{$_} ne $lib) {
      $libroot = $lib;
      $libroot =~ s/lib(.*).a/$1/;
      if ($objdefs{$_} ne "$libroot.o") {
        $DepLibs{$objdefs{$_}} = [] unless exists $DepLibs{$objdefs{$_}};
        push(@{$DepLibs{$objdefs{$_}}}, $_);
      }
    }
  }
  close UNDEFS;
  for my $key (sort keys %DepLibs) {
    if ($FLAT) {
      print " $key";
      if ($WHY) {
        print "\n";
        my @syms = @{$DepLibs{$key}};
        foreach $sym (@syms) {
          print "  $sym\n";
        }
      }
    } else {
      print "    <li>$key</li>\n";
    }
    $suffix = substr($key,length($key)-1,1);
    $key =~ s/(.*)\.[oa]/$1/;
    if ($suffix eq "a") {
      if (!$FLAT) { print DOT "$lib_ns -> $key [ weight=0 ];\n" };
    } else {
      if (!$FLAT) { print DOT "$lib_ns -> $key [ weight=10];\n" };
    }
  }
  if ($FLAT) {
    if (!$WHY) {
      print "\n";
    }
  } else {
    print "  </ul></dd>\n";
  }
}

# Make sure we flush on write. This is slower but correct based on the way we
# write I/O in gen_one_entry.
$| = 1;

# Print the definition list tag
if (!$FLAT) {
    print "<dl>\n";

  open DOT, "| $DotPath -Tgif > libdeps.gif";

  print DOT "digraph LibDeps {size=\"40,15\"; ratio=\"1.33333\"; margin=\"0.25\"; rankdir=\"LR\"; mclimit=\"50.0\"; ordering=\"out\"; center=\"1\";\n";
  print DOT "node [shape=\"box\",color=\"#000088\",fillcolor=\"#FFFACD\",fontcolor=\"#5577DD\",style=\"filled\",fontsize=\"24\"];\n";
  print DOT "edge [style=\"solid\",color=\"#000088\"];\n";
}

# Print libraries first
foreach $lib (@libs) {
  gen_one_entry($lib);
}

if (!$FLAT) {
  print DOT "}\n";
  close DOT;
  open DOT, "| $DotPath -Tgif > objdeps.gif";
  print DOT "digraph ObjDeps {size=\"40,15\"; ratio=\"1.33333\"; margin=\"0.25\"; rankdir=\"LR\"; mclimit=\"50.0\"; ordering=\"out\"; center=\"1\";\n";
  print DOT "node [shape=\"box\",color=\"#000088\",fillcolor=\"#FFFACD\",fontcolor=\"#5577DD\",style=\"filled\",fontsize=\"24\"];\n";
  print DOT "edge [style=\"solid\",color=\"#000088\"];\n";
}

# Print objects second
foreach $obj (@objs) {
  gen_one_entry($obj);
}

if (!$FLAT) {
  print DOT "}\n";
  close DOT;

# Print end tag of definition list element
  print "</dl>\n";
}
