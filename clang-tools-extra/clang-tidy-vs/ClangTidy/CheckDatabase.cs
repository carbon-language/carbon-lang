using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace LLVM.ClangTidy
{
    public class CheckInfo
    {
        [YamlAlias("Name")]
        public string Name { get; set; }

        [YamlAlias("Label")]
        public string Label { get; set; }

        [YamlAlias("Description")]
        public string Desc { get; set; }

        [YamlAlias("Category")]
        public string Category { get; set; }
    }

    /// <summary>
    /// Reads the list of checks from Yaml and builds a description of each one.
    /// This list of checks is then used by the PropertyGrid to determine what
    /// items to display.
    /// </summary>
    public static class CheckDatabase
    {
        static CheckInfo[] Checks_ = null;

        class CheckRoot
        {
            [YamlAlias("Checks")]
            public CheckInfo[] Checks { get; set; }
        }

        static CheckDatabase()
        {
            using (StringReader Reader = new StringReader(Resources.ClangTidyChecks))
            {
                Deserializer D = new Deserializer(namingConvention: new PascalCaseNamingConvention());
                var Root = D.Deserialize<CheckRoot>(Reader);
                Checks_ = Root.Checks;

                HashSet<string> Names = new HashSet<string>();
                foreach (var Check in Checks_)
                {
                    if (Names.Contains(Check.Name))
                        throw new ArgumentException(String.Format("Check {0} exists more than once!", Check.Name));
                    Names.Add(Check.Name);
                }
            }
        }

        public static IEnumerable<CheckInfo> Checks
        {
            get
            {
                return Checks_;
            }
        }
    }
}
