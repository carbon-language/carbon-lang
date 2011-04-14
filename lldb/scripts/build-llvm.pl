#!/usr/bin/perl

# This script will take a number ($ENV{SCRIPT_INPUT_FILE_COUNT}) of static archive files
# and pull them apart into object files. These object files will be placed in a directory
# named the same as the archive itself without the extension. Each object file will then
# get renamed to start with the archive name and a '-' character (for archive.a(object.o)
# the object file would becomde archive-object.o. Then all object files are re-made into
# a single static library. This can help avoid name collisions when different archive
# files might contain object files with the same name.

use strict;
use File::Basename;
use File::Glob ':glob';
use List::Util qw[min max];

our $llvm_srcroot = $ENV{SCRIPT_INPUT_FILE_0};
our $llvm_dstroot = $ENV{SCRIPT_INPUT_FILE_1};

our $libedis_outfile = $ENV{SCRIPT_OUTPUT_FILE_0};
our ($libedis_basename, $libedis_dirname) = fileparse ($libedis_outfile);
our @libedis_slices; # Skinny mach-o slices for libEnhancedDisassembly.dylib

our $llvm_clang_outfile = $ENV{SCRIPT_OUTPUT_FILE_1};
our ($llvm_clang_basename, $llvm_clang_dirname) = fileparse ($llvm_clang_outfile);
our @llvm_clang_slices; # paths to the single architecture static libraries (archives)

our $llvm_configuration = $ENV{LLVM_CONFIGURATION};

our $llvm_revision = "129495";
our $llvm_source_dir = "$ENV{SRCROOT}";
our @archs = split (/\s+/, $ENV{ARCHS});

our @archive_files = (  
    "$llvm_configuration/lib/libclang.a",
	"$llvm_configuration/lib/libclangAnalysis.a",
	"$llvm_configuration/lib/libclangAST.a",
	"$llvm_configuration/lib/libclangBasic.a",
	"$llvm_configuration/lib/libclangCodeGen.a",
	"$llvm_configuration/lib/libclangFrontend.a",
	"$llvm_configuration/lib/libclangDriver.a",
	"$llvm_configuration/lib/libclangIndex.a",
	"$llvm_configuration/lib/libclangLex.a",
	"$llvm_configuration/lib/libclangRewrite.a",
	"$llvm_configuration/lib/libclangParse.a",
	"$llvm_configuration/lib/libclangSema.a",
    "$llvm_configuration/lib/libclangSerialization.a",
	"$llvm_configuration/lib/libCompilerDriver.a",
	"$llvm_configuration/lib/libEnhancedDisassembly.a",
	"$llvm_configuration/lib/libLLVMAnalysis.a",
	"$llvm_configuration/lib/libLLVMArchive.a",
	"$llvm_configuration/lib/libLLVMARMAsmParser.a",
	"$llvm_configuration/lib/libLLVMARMAsmPrinter.a",
	"$llvm_configuration/lib/libLLVMARMCodeGen.a",
	"$llvm_configuration/lib/libLLVMARMDisassembler.a",
	"$llvm_configuration/lib/libLLVMARMInfo.a",
	"$llvm_configuration/lib/libLLVMAsmParser.a",
	"$llvm_configuration/lib/libLLVMAsmPrinter.a",
	"$llvm_configuration/lib/libLLVMBitReader.a",
	"$llvm_configuration/lib/libLLVMBitWriter.a",
	"$llvm_configuration/lib/libLLVMCodeGen.a",
	"$llvm_configuration/lib/libLLVMCore.a",
	"$llvm_configuration/lib/libLLVMExecutionEngine.a",
	"$llvm_configuration/lib/libLLVMInstCombine.a",
	"$llvm_configuration/lib/libLLVMInstrumentation.a",
	"$llvm_configuration/lib/libLLVMipa.a",
	"$llvm_configuration/lib/libLLVMInterpreter.a",
	"$llvm_configuration/lib/libLLVMipo.a",
	"$llvm_configuration/lib/libLLVMJIT.a",
	"$llvm_configuration/lib/libLLVMLinker.a",
	"$llvm_configuration/lib/libLLVMMC.a",
	"$llvm_configuration/lib/libLLVMMCParser.a",
	"$llvm_configuration/lib/libLLVMMCDisassembler.a",
	"$llvm_configuration/lib/libLLVMScalarOpts.a",
	"$llvm_configuration/lib/libLLVMSelectionDAG.a",
	"$llvm_configuration/lib/libLLVMSupport.a",
	"$llvm_configuration/lib/libLLVMTarget.a",
	"$llvm_configuration/lib/libLLVMTransformUtils.a",
	"$llvm_configuration/lib/libLLVMX86AsmParser.a",
	"$llvm_configuration/lib/libLLVMX86AsmPrinter.a",
	"$llvm_configuration/lib/libLLVMX86CodeGen.a",
	"$llvm_configuration/lib/libLLVMX86Disassembler.a",
	"$llvm_configuration/lib/libLLVMX86Info.a",
    "$llvm_configuration/lib/libLLVMX86Utils.a",
);

if (-e "$llvm_srcroot/lib")
{
	print "Using standard LLVM build directory...\n";
	# LLVM in the "lldb" root is a symlink which indicates we are using a 
	# standard LLVM build directory where everything is built into the
	# same folder
	create_single_llvm_arhive_for_arch ($llvm_dstroot, 1);
	my $llvm_dstroot_archive = "$llvm_dstroot/$llvm_clang_basename";
	push @llvm_clang_slices, $llvm_dstroot_archive;
	create_dstroot_file ($llvm_clang_basename, $llvm_clang_dirname, \@llvm_clang_slices, $llvm_clang_basename);
    my $llvm_dstroot_edis = "$llvm_dstroot/$llvm_configuration/lib/libEnhancedDisassembly.dylib";
	if (-f $llvm_dstroot_edis)
	{
		push @libedis_slices, $llvm_dstroot_edis;	
		create_dstroot_file ($libedis_basename, $libedis_dirname, \@libedis_slices, $libedis_basename);	
	} 
	exit 0;
}


if ($ENV{CONFIGURATION} eq "Debug" or $ENV{CONFIGURATION} eq "Release")
{
    # Check for an old llvm source install (not the minimal zip based 
    # install by looking for a .svn file in the llvm directory
    chomp(my $llvm_zip_md5 = `md5 -q $ENV{SRCROOT}/llvm.zip`);
    my $llvm_zip_md5_file = "$ENV{SRCROOT}/llvm/$llvm_zip_md5";
    if (!-e "$llvm_zip_md5_file")
    {
        print "Updating LLVM to use checkpoint from: '$ENV{SRCROOT}/llvm.zip'...\n";
        if (-d "$ENV{SRCROOT}/llvm")
        {
            do_command ("cd '$ENV{SRCROOT}' && rm -rf llvm", "removing old llvm repository", 1);            
        }
		do_command ("cd '$ENV{SRCROOT}' && unzip -q llvm.zip && touch '$llvm_zip_md5_file'", "expanding llvm.zip", 1);
    }

    # We use the stuff in "lldb/llvm" for non B&I builds
    if (!-e $libedis_outfile)
    {
		print "Copying '$ENV{SRCROOT}/llvm/$libedis_basename' to '$libedis_outfile'...\n";
		do_command ("cp '$ENV{SRCROOT}/llvm/$libedis_basename' '$libedis_outfile'", "copying libedis", 1);
    }
    exit 0;
}

# If our output file already exists then we need not generate it again.
if (-e $llvm_clang_outfile and -e $libedis_outfile)
{
	exit 0;
}


# Get our options

our $debug = 1;

sub parallel_guess
{
	my $cpus = `sysctl -n hw.availcpu`;
	chomp ($cpus);
	my $memsize = `sysctl -n hw.memsize`;
	chomp ($memsize);
	my $max_cpus_by_memory = int($memsize / (750 * 1024 * 1024));
	return min($max_cpus_by_memory, $cpus);
}
sub build_llvm
{
	#my $extra_svn_options = $debug ? "" : "--quiet";
	my $svn_options = "--quiet --revision $llvm_revision";
	if (-d "$llvm_source_dir/llvm")
	{
		print "Using existing llvm sources in: '$llvm_source_dir/llvm'\n";
		# print "Updating llvm to revision $llvm_revision\n";
		# do_command ("cd '$llvm_source_dir/llvm' && svn update $svn_options", "updating llvm from repository", 1);
		# print "Updating clang to revision $llvm_revision\n";
		# do_command ("cd '$llvm_source_dir/llvm/tools/clang' && svn update $svn_options", "updating clang from repository", 1);
	}
	else
	{
		print "Checking out llvm sources from revision $llvm_revision...\n";
		do_command ("cd '$llvm_source_dir' && svn co $svn_options http://llvm.org/svn/llvm-project/llvm/trunk llvm", "checking out llvm from repository", 1); 
		print "Checking out clang sources from revision $llvm_revision...\n";
		do_command ("cd '$llvm_source_dir/llvm/tools' && svn co $svn_options http://llvm.org/svn/llvm-project/cfe/trunk clang", "checking out clang from repository", 1);
		print "Removing the llvm/test directory...\n";
		do_command ("cd '$llvm_source_dir' && rm -rf llvm/test", "removing test directory", 1); 
	}

	# Make the llvm build directory
    my $arch_idx = 0;
    foreach my $arch (@archs)
    {
        my $llvm_dstroot_arch = "${llvm_dstroot}/${arch}";

		# if the arch destination root exists we have already built it
		my $do_configure = 0;
		my $do_make = 0;
		
		my $llvm_dstroot_arch_archive = "$llvm_dstroot_arch/$llvm_clang_basename";
		print "LLVM architecture root for ${arch} exists at '$llvm_dstroot_arch'...";
		if (-e $llvm_dstroot_arch)
		{
			print "YES\n";
			$do_configure = !-e "$llvm_dstroot_arch/config.log";
			
			# dstroot for llvm build exists, make sure all .a files are built
			for my $llvm_lib (@archive_files)
			{
				if (!-e "$llvm_dstroot_arch/$llvm_lib")
				{
					print "missing archive: '$llvm_dstroot_arch/$llvm_lib'\n";
					$do_make = 1;
				}
			}	
			if (!-e $llvm_dstroot_arch_archive)
			{
				$do_make = 1;
			}
			else
			{
				print "LLVM architecture archive for ${arch} is '$llvm_dstroot_arch_archive'\n";
			}		
		}
		else
		{
			print "NO\n";
	        do_command ("mkdir -p '$llvm_dstroot_arch'", "making llvm build directory '$llvm_dstroot_arch'", 1);
			$do_configure = 1;
			$do_make = 1;
		}
		
		# If this is the first architecture, then make a symbolic link
		# for any header files that get generated.
	    if ($arch_idx == 0)
 		{
			if (!-l "$llvm_dstroot/llvm")
			{
				do_command ("cd $llvm_dstroot && ln -s './${arch}' llvm");				
			}
		}

		if ($do_configure)
		{
			# Build llvm and clang
	        print "Configuring clang ($arch) in '$llvm_dstroot_arch'...\n";
			my $lldb_configuration_options = '';
			$llvm_configuration eq 'Release' and $lldb_configuration_options .= '--enable-optimized --disable-assertions';
	        do_command ("cd '$llvm_dstroot_arch' && '$llvm_source_dir/llvm/configure' $lldb_configuration_options --enable-targets=x86_64,arm --build=$arch-apple-darwin10",
	                    "configuring llvm build", 1);			
		}

		if ($do_make)
		{
			# Build llvm and clang
			my $num_cpus = parallel_guess();
			print "Building clang using $num_cpus cpus ($arch)...\n";
			do_command ("cd '$llvm_dstroot_arch' && make -j$num_cpus clang-only VERBOSE=1 PROJECT_NAME='llvm'", "making llvm and clang", 1);			
			do_command ("cd '$llvm_dstroot_arch' && make -j$num_cpus tools-only VERBOSE=1 PROJECT_NAME='llvm' EDIS_VERSION=1", "making libedis", 1);			
			# Combine all .o files from a bunch of static libraries from llvm
			# and clang into a single .a file.
			create_single_llvm_arhive_for_arch ($llvm_dstroot_arch, 1);
		}

		-f "$llvm_dstroot_arch_archive" and push @llvm_clang_slices, "$llvm_dstroot_arch_archive";
		-f "$llvm_dstroot_arch/$llvm_configuration/lib/libEnhancedDisassembly.dylib" and push @libedis_slices, "$llvm_dstroot_arch/$llvm_configuration/lib/libEnhancedDisassembly.dylib";
		++$arch_idx;
    }	

    # Combine all skinny slices of the LLVM/Clang combined archive
    create_dstroot_file ($llvm_clang_basename, $llvm_clang_dirname, \@llvm_clang_slices, $llvm_clang_basename);

	if (scalar(@libedis_slices))
	{
		# Combine all skinny slices of the libedis in SYMROOT
		create_dstroot_file ($libedis_basename, $libedis_dirname, \@libedis_slices, $libedis_basename);	

		# Make dSYM for libedis in SYMROOT
		do_command ("cd '$libedis_dirname' && dsymutil $libedis_basename", "making libedis dSYM", 1);			

		# strip debug symbols from libedis and copy into DSTROOT
		-d "$ENV{DSTROOT}/Developer/usr/lib" or do_command ("mkdir -p '$ENV{DSTROOT}/Developer/usr/lib'", "Making directory '$ENV{DSTROOT}/Developer/usr/lib'", 1);
		do_command ("cd '$libedis_dirname' && strip -Sx -o '$ENV{DSTROOT}/Developer/usr/lib/$libedis_basename' '$libedis_outfile'", "Stripping libedis and copying to DSTROOT", 1);			
	}
}

sub create_dstroot_file
{
	my $file = shift;
	my $dir = shift;
	my $fullpath = "$dir/$file";	# The path to the file to create
	my $slice_aref = shift; # Array containing one or more skinny files that will be combined into $fullpath
	my $what = shift;		# Text describing the $fullpath

    print "create_dstroot_file file = '$file', dir = '$dir', slices = (" . join (', ', @$slice_aref) . ") for what = '$what'\n";

	if (-d $dir)
	{
		if (@$slice_aref > 0)
		{
			print "Creating and installing $what into '$fullpath'...\n";
			my $lipo_command = "lipo -output '$fullpath' -create";
			foreach (@$slice_aref) { $lipo_command .= " '$_'"; }
			do_command ($lipo_command, "creating $what universal output file", 1);
		}
	

		if (!-e $fullpath)
		{
			# die "error: '$fullpath' is missing\n";
		}
	}
	else
	{
		die "error: directory '$dir' doesn't exist to receive file '$file'\n";
	}
}
#----------------------------------------------------------------------
# quote the path if needed and realpath it if the -r option was 
# specified
#----------------------------------------------------------------------
sub finalize_path
{
	my $path = shift;
	# Realpath all paths that don't start with "/"
	$path =~ /^[^\/]/ and $path = abs_path($path);

	# Quote the path if asked to, or if there are special shell characters
	# in the path name
	my $has_double_quotes = $path =~ /["]/;
	my $has_single_quotes = $path =~ /[']/;
	my $needs_quotes = $path =~ /[ \$\&\*'"]/;
	if ($needs_quotes)
	{
		# escape and double quotes in the path
		$has_double_quotes and $path =~ s/"/\\"/g;
		$path = "\"$path\"";
	}
	return $path;
}

sub do_command
{
	my $cmd = shift;
	my $description = @_ ? shift : "command";
	my $die_on_fail = @_ ? shift : undef;
	$debug and print "% $cmd\n";
	system ($cmd);
	if ($? == -1) 
	{
        $debug and printf ("error: %s failed to execute: $!\n", $description);
		$die_on_fail and $? and exit(1);
		return $?;
    }
    elsif ($? & 127) 
	{
        $debug and printf("error: %s child died with signal %d, %s coredump\n", 
						  $description, 
						  ($? & 127),  
						  ($? & 128) ? 'with' : 'without');
		$die_on_fail and $? and exit(1);
		return $?;
    }
    else 
	{
		my $exit = $? >> 8;
		if ($exit)
		{
			$debug and printf("error: %s child exited with value %d\n", $description, $exit);
			$die_on_fail and exit(1);
		}
		return $exit;
    }
}

sub create_single_llvm_arhive_for_arch
{
	my $arch_dstroot = shift;
    my $split_into_objects = shift;
	my @object_dirs;
	my $object_dir;
	my $tmp_dir = $arch_dstroot;
	my $arch_output_file = "$arch_dstroot/$llvm_clang_basename";
    -e $arch_output_file and return;
	my $files = "$arch_dstroot/files.txt";
	open (FILES, ">$files") or die "Can't open $! for writing...\n";

	for my $path (@archive_files) 
	{
		my $archive_fullpath = finalize_path ("$arch_dstroot/$path");
		if (-e $archive_fullpath)
		{
            if ($split_into_objects)
            {
                my ($archive_file, $archive_dir, $archive_ext) = fileparse($archive_fullpath, ('.a'));
        
                $object_dir = "$tmp_dir/$archive_file";
                push @object_dirs, $object_dir;
            
                do_command ("cd '$tmp_dir'; mkdir '$archive_file'; cd '$archive_file'; ar -x $archive_fullpath");
        
                my @objects = bsd_glob("$object_dir/*.o");
        
                foreach my $object (@objects)
                {
                    my ($o_file, $o_dir) = fileparse($object);
                    my $new_object = "$object_dir/${archive_file}-$o_file";
                    print FILES "$new_object\n";
                    do_command ("mv '$object' '$new_object'");
                }				
            }
            else
            {
                # just add the .a files into the file list
                print FILES "$archive_fullpath\n";
            }
		}
        else
        {
            print "warning: archive doesn't exist: '$archive_fullpath'\n";
        }
	}
	close (FILES);
    do_command ("libtool -static -o '$arch_output_file' -filelist '$files'");
    do_command ("ranlib '$arch_output_file'");

	foreach $object_dir (@object_dirs)
	{
		do_command ("rm -rf '$object_dir'");
	}
	do_command ("rm -rf '$files'");
}

build_llvm();
