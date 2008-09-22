import BaseHTTPServer
import SimpleHTTPServer
import os
import sys
import urllib, urlparse
import posixpath
import StringIO
import re
import shutil
import threading
import time
import socket

import Reporter

# Keys replaced by server.

kReportColRE = re.compile('<!-- REPORTBUGCOL -->')
kReportColRepl = '<td></td>'
kReportBugRE = re.compile('<!-- REPORTBUG id="report-(.*)\\.html" -->')         
kReportBugRepl = '<td class="ReportBug"><a href="report/\\1">Report Bug</a></td>'
kBugKeyValueRE = re.compile('<!-- BUG([^ ]*) (.*) -->')

kReportReplacements = [(kReportColRE, kReportColRepl),
                       (kReportBugRE, kReportBugRepl)]

# Other simple parameters

kResources = posixpath.join(posixpath.dirname(__file__), 'Resources')

###

__version__ = "0.1"

__all__ = ["create_server"]

class ReporterThread(threading.Thread):
    def __init__(self, report, reporter, parameters, server):
        threading.Thread.__init__(self)
        self.report = report
        self.server = server
        self.reporter = reporter
        self.parameters = parameters
        self.status = None

    def run(self):
        result = None
        try:
            if self.server.options.debug:
                print >>sys.stderr, "%s: SERVER: submitting bug."%(sys.argv[0],)
            result = self.reporter.fileReport(self.report, self.parameters)
            time.sleep(3)
            if self.server.options.debug:
                print >>sys.stderr, "%s: SERVER: submission complete."%(sys.argv[0],)
        except Reporter.ReportFailure,e:
            s = StringIO.StringIO()
            print >>s,'Submission Failed<br><pre>'
            print >>s,e.value
            print >>s,'</pre>'
            self.status = s.getvalue()
            return            
        except Exception,e:
            s = StringIO.StringIO()
            import traceback
            print >>s,'Submission Failed<br><pre>'
            traceback.print_exc(e,file=s)
            print >>s,'</pre>'
            self.status = s.getvalue()
            return

        s = StringIO.StringIO()
        print >>s, 'Submission Complete!'
        if result is not None:
            print >>s, '<hr>'
            print >>s, result
        self.status = s.getvalue()

class ScanViewServer(BaseHTTPServer.HTTPServer):
    def __init__(self, address, handler, root, reporters, options):
        BaseHTTPServer.HTTPServer.__init__(self, address, handler)
        self.root = root
        self.reporters = reporters
        self.options = options        
        self.halted = False

    def halt(self):
        self.halted = True
        if self.options.debug:
            print >>sys.stderr, "%s: SERVER: halting." % (sys.argv[0],)

    def serve_forever(self):
        while not self.halted:
            if self.options.debug > 1:
                print >>sys.stderr, "%s: SERVER: waiting..." % (sys.argv[0],)
            try:
                self.handle_request()
            except OSError,e:
                print 'OSError',e.errno

    def finish_request(self, request, client_address):
        if self.options.autoReload:
            import ScanView
            self.RequestHandlerClass = reload(ScanView).ScanViewRequestHandler
        BaseHTTPServer.HTTPServer.finish_request(self, request, client_address)

    def handle_error(self, request, client_address):
        # Ignore socket errors
        info = sys.exc_info()
        if info and isinstance(info[1], socket.error):
            if self.options.debug > 1:
                print >>sys.stderr, "%s: SERVER: ignored socket error." % (sys.argv[0],)
            return
        BaseHTTPServer.HTTPServer.handle_error(self, request, client_address)

# Borrowed from Quixote, with simplifications.
def parse_query(qs, fields=None):
    if fields is None:
        fields = {}
    for chunk in filter(None, qs.split('&')):
        if '=' not in chunk:
            name = chunk
            value = ''
        else:
            name, value = chunk.split('=', 1)
        name = urllib.unquote(name.replace('+', ' '))
        value = urllib.unquote(value.replace('+', ' '))
        fields[name] = value
    return fields

class ScanViewRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    server_version = "ScanViewServer/" + __version__
    dynamic_mtime = time.time()

    def do_HEAD(self):
        try:
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_HEAD(self)
        except Exception,e:
            self.handle_exception(e)
            
    def do_GET(self):
        try:
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
        except Exception,e:
            self.handle_exception(e)
            
    def do_POST(self):
        """Serve a POST request."""
        try:
            length = self.headers.getheader('content-length') or "0"
            try:
                length = int(length)
            except:
                length = 0
            content = self.rfile.read(length)
            fields = parse_query(content)
            f = self.send_head(fields)
            if f:
                self.copyfile(f, self.wfile)
                f.close()
        except Exception,e:
            self.handle_exception(e)            

    def log_message(self, format, *args):
        if self.server.options.debug:
            sys.stderr.write("%s: SERVER: %s - - [%s] %s\n" %
                             (sys.argv[0],
                              self.address_string(),
                              self.log_date_time_string(),
                              format%args))

    def load_report(self, report):
        path = os.path.join(self.server.root, 'report-%s.html'%report)
        data = open(path).read()
        keys = {}
        for item in kBugKeyValueRE.finditer(data):
            k,v = item.groups()
            keys[k] = v
        return keys

    def handle_exception(self, exc):
        import traceback
        s = StringIO.StringIO()
        print >>s, "INTERNAL ERROR\n"
        traceback.print_exc(exc, s)
        f = self.send_string(s.getvalue(), 'text/plain')
        if f:
            self.copyfile(f, self.wfile)
            f.close()        
    
    def send_internal_error(self, message):
        return self.send_string('ERROR: %s'%(message,), 'text/plain')

    def send_report_submit(self):
        s = StringIO.StringIO()
        report = self.fields.get('report')
        reporter = self.fields.get('reporter')
        title = self.fields.get('title')
        description = self.fields.get('description')
        
        # Get the reporter and parameters.
        reporter = self.server.reporters[int(reporter)]
        parameters = {}
        for o in reporter.getParameterNames():
            name = '%s_%s'%(reporter.getName(),o)
            parameters[o] = self.fields.get(name)

        # Create the report.
        path = os.path.join(self.server.root, 'report-%s.html'%report)
        files = [path]
        br = Reporter.BugReport(title, description, files)

        # Send back an initial response and wait for the report to
        # finish.
        initial_response = """<html>
<head>
  <title>Filing Report</title>
  <link rel="stylesheet" type="text/css" href="/scanview.css" />
</head>
<body>
<h1>Filing Report</h1>
<b>Report</b>: %(report)s<br>
<b>Title</b>: %(title)s<br>
<b>Description</b>: %(description)s<br>
<hr>
Submission in progress."""%locals()

        self.send_response(200)
        self.send_header("Content-type", 'text/html')
        self.end_headers()
        self.wfile.write(initial_response)
        self.wfile.flush()
        
        # Kick off a reporting thread.
        t = ReporterThread(br, reporter, parameters, self.server)
        t.start()

        # Wait for thread to die...
        while t.isAlive():
            self.wfile.write('.')
            self.wfile.flush()
            time.sleep(.25)
        submitStatus = t.status

        end_response = """<br>
%(submitStatus)s
<hr>
<a href="/">Home</a>
</body>
</html>
"""%locals()
        return self.send_string(end_response, headers=False)

    def send_report(self, report):
        try:
            keys = self.load_report(report)
        except IOError:
            return self.send_internal_error('Invalid report.')

        initialTitle = keys.get('DESC','')
        initialDescription = """\
Bug generated by the clang static analyzer.

Description: %s
File: %s
Line: %s
"""%(initialTitle,
     keys.get('FILE','<unknown>'),
     keys.get('LINE','<unknown>'))

        reporterSelections = []
        reporterOptions = []
        
        for i,r in enumerate(self.server.reporters):
            reporterSelections.append('<option value="%d">%s</option>'%(i,r.getName()))
            options = '\n'.join(["""\
<tr>
  <td class="form_clabel">%s:</td>
  <td class="form_value"><input type="text" name="%s_%s"></td>
</tr>"""%(o,r.getName(),o) for o in r.getParameterNames()])
            if i==0:
                display = ''
            else:
                display = 'none'
            reporterOptions.append("""\
<tr id="%sReporterOptions" style="display:%s">
  <td class="form_label">%s Options</td>
  <td class="form_value">
    <table class="form_inner_group">
%s
    </table>
  </td>
</tr>
"""%(r.getName(),display,r.getName(),options))
        reporterSelections = '\n'.join(reporterSelections)
        reporterOptionsDivs = '\n'.join(reporterOptions)
        reportersArray = '[%s]'%(','.join([`r.getName()` for r in self.server.reporters]))

        result = """<html>
<head>
  <title>File Report</title>
  <link rel="stylesheet" type="text/css" href="/scanview.css" />
</head>
<script language="javascript" type="text/javascript">
var reporters = %(reportersArray)s;
function updateReporterOptions() {
  index = document.getElementById('reporter').selectedIndex;
  for (var i=0; i < reporters.length; ++i) {
    o = document.getElementById(reporters[i] + "ReporterOptions");
    if (i == index) {
      o.style.display = "";
    } else {
      o.style.display = "none";
    }
  }
}
</script>
<body>
<h1>File Report</h1>
<form name="form" action="/report_submit" method="post">
<input type="hidden" name="report" value="%(report)s">

<table class="form">
<tr><td>
<table class="form_group">
<tr>
  <td class="form_clabel">Title:</td>
  <td class="form_value">
    <input type="text" name="title" size="50" value="%(initialTitle)s">
  </td>
</tr>
<tr>
  <td class="form_label">Description:</td>
  <td class="form_value">
<textarea rows="10" cols="80" name="description">
%(initialDescription)s
</textarea>
  </td>
</table>
<br>
<table class="form_group">
<tr>
  <td class="form_clabel">Method:</td>
  <td class="form_value">
    <select id="reporter" name="reporter" onChange="updateReporterOptions()">
    %(reporterSelections)s
    </select>
  </td>
</tr>
%(reporterOptionsDivs)s
</table>
<br>
</td></tr>
<tr><td class="form_submit">
  <input align="right" type="submit" name="Submit" value="Submit">
</td></tr>
</table>
</form>


<iframe src="/report-%(report)s.html#EndPath" width="100%%" height="40%%"
        scrolling="auto" frameborder="1">
  <a href="/report-%(report)s.html#EndPath">View Bug Report</a>
</iframe>

</body>
</html>"""%locals()
        return self.send_string(result)

    def send_head(self, fields=None):
        if fields is None:
            fields = {}
        self.fields = fields

        o = urlparse.urlparse(self.path)
        self.fields = parse_query(o.query, fields)
        path = posixpath.normpath(urllib.unquote(o.path))

        # Split the components and strip the root prefix.
        components = path.split('/')[1:]
        
        # Special case some top-level entries.
        if components:
            name = components[0]
            if len(components)==2:
                if name=='report':
                    return self.send_report(components[1])
            elif len(components)==1:
                if name=='quit':
                    self.server.halt()
                    return self.send_string('Goodbye.', 'text/plain')
                elif name=='report_submit':
                    return self.send_report_submit()
                elif name=='favicon.ico':
                    return self.send_path(posixpath.join(kResources,'bugcatcher.ico'))
                elif name=='scanview.css':
                    return self.send_path(posixpath.join(kResources,'scanview.css'))
        
        # Match directory entries.
        if components[-1] == '':
            components[-1] = 'index.html'

        suffix = '/'.join(components)

        # The summary may reference source files on disk using rooted
        # paths. Make sure these resolve correctly for now.
        # FIXME: This isn't a very good idea... we should probably
        # mark rooted paths somehow.        
        if os.path.exists(posixpath.join('/', suffix)):
            path = posixpath.join('/', suffix)
        else:
            path = posixpath.join(self.server.root, suffix)

        if self.server.options.debug > 1:
            print >>sys.stderr, '%s: SERVER: sending path "%s"'%(sys.argv[0],
                                                                 path)
        return self.send_path(path)

    def send_404(self):
        self.send_error(404, "File not found")
        return None

    def send_path(self, path):
        ctype = self.guess_type(path)
        if ctype.startswith('text/'):
            # Patch file instead
            return self.send_patched_file(path, ctype)
        else:
            mode = 'rb'
        try:
            f = open(path, mode)
        except IOError:
            return self.send_404()
        return self.send_file(f, ctype)

    def send_file(self, f, ctype):
        # Patch files to add links, but skip binary files.
        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def send_string(self, s, ctype='text/html', headers=True, mtime=None):
        if headers:
            self.send_response(200)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(len(s)))
            if mtime is None:
                mtime = self.dynamic_mtime
            self.send_header("Last-Modified", self.date_time_string(mtime))
            self.end_headers()
        return StringIO.StringIO(s)

    def send_patched_file(self, path, ctype):
        f = open(path,'r')
        fs = os.fstat(f.fileno())
        data = f.read()
        for a,b in kReportReplacements:
            data = a.sub(b, data)
        return self.send_string(data, ctype, mtime=fs.st_mtime)


def create_server(options, root):
    import Reporter

    reporters = Reporter.getReporters()

    return ScanViewServer((options.host, options.port),
                          ScanViewRequestHandler,
                          root,
                          reporters,
                          options)
