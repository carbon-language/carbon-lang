<?php

/**
 * llvm-lit wrapper
 *
 * To use, set unit.engine in .arcconfig, or use --engine flag
 * with arc unit.
 *
 * This file was authored by Clemens Hammacher <hammacher@cs.uni-saarland.de>
 * and initially used in the Sambamba project (www.sambamba.org).
 *
 * @group unitrun
 */
final class LitTestEngine extends ArcanistUnitTestEngine {

    protected function supportsRunAllTests() {
        return true;
    }

    private function progress($results, $numTests) {
        static $colors = array(
            ArcanistUnitTestResult::RESULT_PASS => 'green',
            ArcanistUnitTestResult::RESULT_FAIL => 'red',
            ArcanistUnitTestResult::RESULT_SKIP => 'yellow',
            ArcanistUnitTestResult::RESULT_BROKEN => 'red',
            ArcanistUnitTestResult::RESULT_UNSOUND => 'yellow',
            ArcanistUnitTestResult::RESULT_POSTPONED => 'yellow'
        );

        $s = "\t[\033[0;30m";
        $number = -1;
        $lastColor = "";
        $lastResult = "";
        $lastNumber = "";
        foreach ($results as $result) {
            $color = $colors[$result->getResult()];
            if (!$color && $lastColor)
                $s .= "</bg>";
            elseif (!$lastColor && $color)
                $s .= "<bg:$color>";
            elseif ($lastColor !== $color)
                $s .= "</bg><bg:$color>";
            if ($number <= 0)
              $number = 1;
            elseif ($lastResult == $result->getResult())
              $number += 1;
            else
              $number = 1;
            if ($number > 1 && $number <= 10)
              $s = substr($s, 0, -1);
            elseif ($number > 10 && $number <= 100)
              $s = substr($s, 0, -2);
            elseif ($number > 100 && $number <= 1000)
              $s = substr($s, 0, -3);
            $s .= "$number";
            $lastNumber = $number;
            $lastResult = $result->getResult();
            $lastColor = $color;
        }
        if ($lastColor)
            $s .= "</bg>";
        $s .= "\033[0m";
        $c = count($results);
        if ($numTests)
          $s .=  " $c/$numTests]";
        else
            $s .= " $c]";
        return phutil_console_format($s);
    }

    public function run() {

        $projectRoot = $this->getWorkingCopy()->getProjectRoot();
        $cwd = getcwd();
        $buildDir = $this->findBuildDirectory($projectRoot, $cwd);
        $pollyObjDir = $buildDir;
        if (is_dir($buildDir.DIRECTORY_SEPARATOR."tools".DIRECTORY_SEPARATOR."polly"))
          $pollyObjDir = $buildDir.DIRECTORY_SEPARATOR."tools".DIRECTORY_SEPARATOR."polly";
        $pollyTestDir = $pollyObjDir.DIRECTORY_SEPARATOR."test";

        if (is_dir($buildDir.DIRECTORY_SEPARATOR."bin") &&
          file_exists($buildDir.DIRECTORY_SEPARATOR."bin".DIRECTORY_SEPARATOR."llvm-lit")) {
          $lit = $buildDir.DIRECTORY_SEPARATOR."bin".DIRECTORY_SEPARATOR."llvm-lit";
          $cmd = "ninja -C ".$buildDir;
          print "Running ninja (".$cmd.")\n";
          exec($cmd);
        } else {
          $makeVars = $this->getMakeVars($buildDir);
          $lit = $this->findLitExecutable($makeVars);
        }
        print "Using lit executable '$lit'\n";

        // We have to modify the format string, because llvm-lit does not like a '' argument
        $cmd = '%s ' . ($this->getEnableAsyncTests() ? '' : '-j1 ') .'%s 2>&1';
        $litFuture = new ExecFuture($cmd, $lit, $pollyTestDir);
        $out = "";
        $results = array();
        $lastTime = microtime(true);
        $ready = false;
        $dots = "";
        $numTests = 0;
        while (!$ready) {
            $ready = $litFuture->isReady();
            $newout = $litFuture->readStdout();
            if (strlen($newout) == 0) {
                usleep(100);
                continue;
            }
            $out .= $newout;
            if ($ready && strlen($out) > 0 && substr($out, -1) != "\n")
                $out .= "\n";

            while (($nlPos = strpos($out, "\n")) !== FALSE) {
                $line = substr($out, 0, $nlPos+1);
                $out = substr($out, $nlPos+1);

                $res = ArcanistUnitTestResult::RESULT_UNSOUND;
                if (substr($line, 0, 6) == "PASS: ") {
                    $res = ArcanistUnitTestResult::RESULT_PASS;
                } elseif (substr($line, 0, 6) == "FAIL: ") {
                    $res = ArcanistUnitTestResult::RESULT_FAIL;
                } elseif (substr($line, 0, 7) == "XPASS: ") {
                    $res = ArcanistUnitTestResult::RESULT_FAIL;
                } elseif (substr($line, 0, 7) == "XFAIL: ") {
                    $res = ArcanistUnitTestResult::RESULT_PASS;
                } elseif (substr($line, 0, 13) == "UNSUPPORTED: ") {
                    $res = ArcanistUnitTestResult::RESULT_SKIP;
                } elseif (!$numTests && preg_match('/Testing: ([0-9]+) tests/', $line, $matches)) {
                    $numTests = (int)$matches[1];
                }
                if ($res == ArcanistUnitTestResult::RESULT_FAIL)
                    print "\033[0A";
                if ($res != ArcanistUnitTestResult::RESULT_SKIP && $res != ArcanistUnitTestResult::RESULT_PASS)
                    print "\r\033[K\033[0A".$line.self::progress($results, $numTests);
                if ($res == ArcanistUnitTestResult::RESULT_UNSOUND)
                    continue;
                $result = new ArcanistUnitTestResult();
                $result->setName(trim(substr($line, strpos($line, ':') + 1)));
                $result->setResult($res);
                $newTime = microtime(true);
                $result->setDuration($newTime - $lastTime);
                $lastTime = $newTime;
                $results[] = $result;
                $dots .= ".";
                print "\r\033[K\033[0A".self::progress($results, $numTests);
            }
        }
        list ($out1,$out2) = $litFuture->read();
        print $out1;
        if ($out2) {
            throw new Exception('There was error output, even though it should have been redirected to stdout.');
        }
        print "\n";

        $timeThreshold = 0.050;
        $interestingTests = array();
        foreach ($results as $result) {
          if ($result->getResult() != "pass")
            $interestingTests[] = $result;
          if ($result->getDuration() > $timeThreshold)
            $interestingTests[] = $result;
        }
        return $interestingTests;
    }

    /**
     * Try to find the build directory of the project, starting from the
     * project root, and the current working directory.
     * Also, the environment variable POLLY_BIN_DIR is read to determine the
     * correct location.
     *
     * Search locations are:
     * $POLLY_BIN_DIR (environment variable)
     * <root>/build
     * <root>.build
     * <root>-build
     * <root:s/src/build>
     * <cwd>
     *
     * This list might be extended in the future according to other common
     * setup.
     *
     * @param   projectRoot   Directory of the project root
     * @param   cwd           Current working directory
     * @return  string        Presumable build directory
     */
    public static function findBuildDirectory($projectRoot, $cwd) {

        $projectRoot = rtrim($projectRoot, DIRECTORY_SEPARATOR);
        $cwd = rtrim($cwd, DIRECTORY_SEPARATOR);

        $tries = array();

        $smbbin_env = getenv("POLLY_BIN_DIR");
        if ($smbbin_env)
            $tries[] = $smbbin_env;

        $tries[] = $projectRoot.DIRECTORY_SEPARATOR."build";
        $tries[] = $projectRoot.".build";
        $tries[] = $projectRoot."-build";

        // Try to replace each occurence of "src" by "build" (also within path components, like llvm-src)
        $srcPos = 0;
        while (($srcPos = strpos($projectRoot, "src", $srcPos)) !== FALSE) {
            $tries[] = substr($projectRoot, 0, $srcPos)."build".substr($projectRoot, $srcPos+3);
            $srcPos += 3;
        }

        $tries[] = $cwd;

        foreach ($tries as $try) {
            if (is_dir($try) &&
                (file_exists($try.DIRECTORY_SEPARATOR."test".DIRECTORY_SEPARATOR."lit.site.cfg") ||
                file_exists($try.DIRECTORY_SEPARATOR."test".DIRECTORY_SEPARATOR."lit.cfg")))
                return Filesystem::resolvePath($try);
        }

        throw new Exception("Did not find a build directory for project '$projectRoot', cwd '$cwd'.\n" .
            "Make sure to have a 'test' directory inside build, which contains lit.cfg or lit.site.cfg.\n" .
            "You might have to run 'make test' once in the build directory.");
    }

    /**
     * Try to find the llvm build directory, based on the make variables
     * as determined by getMakeVars().
     *
     * @param   makeVars      The determined make variables
     * @return  string        Presumable llvm build directory
     */
    public static function getLLVMObjDir($makeVars) {
        if (!array_key_exists('LLVM_OBJ_ROOT', $makeVars))
            throw new Exception("Make variables (determined by 'make printvars') does not contain LLVM_OBJ_ROOT.");
        $llvmObjDir = $makeVars['LLVM_OBJ_ROOT'];
        if (!is_dir($llvmObjDir))
            throw new Exception("LLVM_OBJ_ROOT ('$llvmObjDir') is no directory.");
        return $llvmObjDir;
    }

    /**
     * Determine the make variables, by calling 'make printvars' in the
     * build directory.
     *
     * @param   buildDir      The determined build directory
     * @return  map<str,str>  Map of printed make variables to values
     */
    public static function getMakeVars($buildDir) {
        $litFuture = new ExecFuture('make -C %s printvars', $buildDir);
        list($stdout, $stderr) = $litFuture->resolvex(10);
        print $stderr;
        $makeVars = array();

        foreach (explode("\n", $stdout) as $line) {
            $components = explode(':', $line);
            if (count($components) == 3)
                $makeVars[trim($components[1])] = trim($components[2]);
        }

        return $makeVars;
    }

    /**
     * Return full path to the llvm-lit executable.
     *
     * @param   llvmObjDir    The determined llvm build directory
     * @return  string        Full path to the llvm-lit executable
     */
    public static function findLitExecutable($makeVars) {
        $llvmObjDir = self::getLLVMObjDir($makeVars);
        $buildMode = array_key_exists('BuildMode', $makeVars) ? $makeVars['BuildMode'] : '';

        if (!$buildMode)
            throw new Exception("Make variables (determined by 'make printvars') does not contain BuildMode.");

        $lit = $llvmObjDir.DIRECTORY_SEPARATOR.$buildMode.DIRECTORY_SEPARATOR.'bin'.DIRECTORY_SEPARATOR.'llvm-lit';
        if (!is_executable($lit))
            throw new Exception("File does not exists or is not executable: $lit");
        return $lit;
    }
}
