#!/usr/bin/perl -w
#
# Program:  GenLibDeps.pl
#
# Synopsis: Generate HTML output that shows the dependencies between a set of
#           libraries. The output of this script should periodically replace 
#           the similar content in the UsingLibraries.html document.
#
# Syntax:   GenLibDeps.pl <directory_with_libraries_in_it>
#

# Give first option a name.
my $Directory = $ARGV[0];

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
    "nm -g --defined-only $lib | grep ' [ABCDGRST] ' | sed -e 's/^[0-9A-Fa-f]* [ABCDGRST] //' | sort | uniq |";
  while (<DEFS>) {
    chomp($_);
    $libdefs{$_} = $lib;
  }
  close DEFS;
}

# Gather definitions from the object files.
foreach $obj (@objs ) {
  open DEFS, 
    "nm -g --defined-only $obj | grep ' [ABCDGRST] ' | sed -e 's/^[0-9A-Fa-f]* [ABCDGRST] //' | sort | uniq |";
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
  print "  <dt><b>$lib</b</dt><dd><ul>\n";
  open UNDEFS, 
    "nm -u $lib | grep ' U ' | sed -e 's/         U //' | sort | uniq |";
  open DEPENDS,
    "| sort | uniq > GenLibDeps.out";
  while (<UNDEFS>) {
    chomp;
    if (defined($libdefs{$_}) && $libdefs{$_} ne $lib) {
      print DEPENDS "$libdefs{$_}\n";
    } elsif (defined($objdefs{$_}) && $objdefs{$_} ne $lib) {
      $libroot = $lib;
      $libroot =~ s/lib(.*).a/$1/;
      if ($objdefs{$_} ne "$libroot.o") {
        print DEPENDS "$objdefs{$_}\n";
      }
    }
  }
  close UNDEFS;
  close DEPENDS;
  open DF, "<GenLibDeps.out";
  while (<DF>) {
    chomp;
    print "    <li>$_</li>\n";
  }
  close DF;
  print "  </ul></dd>\n";
}

# Make sure we flush on write. This is slower but correct based on the way we
# write I/O in gen_one_entry.
$| = 1;

# Print the definition list tag
print "<dl>\n";

# Print libraries first
foreach $lib (@libs) {
  gen_one_entry($lib);
}

# Print objects second
foreach $obj (@objs) {
  gen_one_entry($obj);
}

# Print end tag of definition list element
print "</dl>\n";
