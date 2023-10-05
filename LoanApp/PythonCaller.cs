using Newtonsoft.Json;
using System.Diagnostics;
using System.Collections.Generic;
using System;

namespace LoanApp
{
    class PythonCallerArgs
    {
        private Dictionary<string, object> modelArgs;
        private Dictionary<string, object> metaArgs;
        private JsonSerializerSettings jsonSer;

        public PythonCallerArgs()
        {
            modelArgs = new Dictionary<string, object>();
            metaArgs = new Dictionary<string, object>();
            jsonSer = new JsonSerializerSettings();
            jsonSer.StringEscapeHandling = StringEscapeHandling.EscapeHtml;
        }

        public void AddMetaArg(string property, object obj)
        {
            metaArgs.Add(JsonConvert.SerializeObject(property, jsonSer), JsonConvert.SerializeObject(obj, jsonSer));
        }

        public void AddArg(string property, object obj)
        {
            modelArgs.Add(property, obj);
        }
        public string Serialized()
        {
            DateTime tm = DateTime.Now;
            AddMetaArg("date", tm);
            AddMetaArg("model_input", modelArgs);
            return JsonConvert.SerializeObject(metaArgs);
        }
    }
    
    //Class for script caller.
    class PythonCaller
    {
        private ProcessStartInfo psi;
        private string scriptPath = "";
        private string scriptName = "";
        private string errors = "";
        private string results = "";
        private string caller = "";
        private string envPath = "";

        public PythonCaller(string scriptPath_, string scriptName_)
        {
            caller = System.IO.File.ReadAllText("callerPath.txt") + "\\caller_script\\caller.py";

            psi = new ProcessStartInfo();
            psi.FileName = System.IO.File.ReadAllText("pythonPath.txt");
            string[] separatingStrings = { "\\python.exe"};
            envPath = psi.FileName.Split(separatingStrings, System.StringSplitOptions.RemoveEmptyEntries)[0];
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;

            scriptPath = scriptPath_;
            scriptName = scriptName_;
        }

        public Dictionary<string, string> CallClassMethod(string className, string method, PythonCallerArgs args)
        {
            results = "";
            errors = "";
            string argString = args.Serialized();

            try
            {
                psi.Arguments =  $"\"{caller}\" \"{scriptPath}\" \"{scriptName}\" \"{className}\" \"{method}\" \"{argString}\" \"{System.IO.File.ReadAllText("callerPath.txt")}\"";
                

                var value = System.Environment.GetEnvironmentVariable("PATH");
                var new_value = envPath + "\\:" + envPath + "\\Scripts:" + value;
                System.Environment.SetEnvironmentVariable("PATH", new_value);

                using (var process = Process.Start(psi))
                {
                    errors = process.StandardError.ReadToEnd();
                    results = process.StandardOutput.ReadToEnd();
                    
                }
                Console.WriteLine(errors);
                Console.WriteLine(results);
                Dictionary<string, string> ob = JsonConvert.DeserializeObject<Dictionary<string, string>>(results);
                return ob;
            }catch (Exception ex)
            {
                errors = ex.Message;
            }
            return null;
        }

        public string GetErrors()
        {
            return errors;
        }
    }
}
