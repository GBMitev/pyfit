# %%
from .dependencies import *

def write_fit(input_file, params_dict, index_dict):
    if index_dict is None:
        index_dict = {key: val[1] for key, val in params_dict.items()}

    path = os.path.abspath(input_file)
    dir,fname = os.path.split(path)
    
    oname = f"FIT_{fname}"
    opath = os.path.join(dir, oname)

    fpath = os.path.join(dir, fname)

    with open(f"{fpath}", "r") as file:
        lines = file.readlines()

    for key, val in params_dict.items():
        parameter = key.split("_")[-1]
        value     = f"{val.value}"

        new_line = f"\t {parameter} \t\t {value}\n"
        line_num = index_dict[key]

        lines[line_num] = new_line
    
    lines[0] += "\t\t (This file has been produced using PyFit)"
    with open(f"{opath}", "w") as file:
        file.writelines(lines)
    
    return opath

def run_Iteration(input_file,duo_command,states_file_name,params_dict, index_dict):

    input_file = write_fit(input_file, params_dict, index_dict)
    
    command = f"{duo_command} <{input_file}> output.out"
    running = sp.Popen(command, stdout=sp.PIPE, shell = True)
    running.communicate()
    
    column_names = ["NN","E","gns","J","tau","e/f","Manifold","v","Lambda","Sigma","Omega"]

    df = pd.read_csv(f"./{states_file_name}.states", delim_whitespace = True,names = column_names)
    
    rm_commands = ["mkdir waste", f"mv {states_file_name}.states waste",f"mv {states_file_name}.trans waste", "mv output.out waste"]

    current_directory = os.getcwd()
    full_path = os.path.join(current_directory, "waste")
    if os.path.exists(full_path) and os.path.isdir(full_path):
        rm_commands = rm_commands[1:]

    for cmd in rm_commands:    
        process = sp.Popen(cmd, shell = True, stdout = sp.PIPE, text = True)
        _, _ = process.communicate()
    
    tau_cond = [df["tau"] == "+",df["tau"] == "-"]
    tau_vals = [1,-1]

    df["tau"] = np.select(tau_cond, tau_vals)

    return df

def get_residuals(iter, marvel, Q_nums, Energy_handle):
    iter = iter[Q_nums+[Energy_handle]]
    marvel = marvel[Q_nums+[Energy_handle]]
    
    OBS = Energy_handle+"_OBS"
    CALC = Energy_handle+"_CALC"

    merged = marvel.merge(iter, on = Q_nums, how = "inner",suffixes = ["_OBS","_CALC"])
    merged["OC"] = merged[OBS]-merged[CALC]

    merged = merged[Q_nums+[OBS,CALC,"OC"]]

    OC = merged["OC"].to_numpy()
    rms = np.sqrt(np.sum(OC**2)/len(OC))

    return rms, merged

class Duo_Input:
    def __init__(self, input_file, extra_cards = None):
        self.input_file = input_file

        def read_in_values(ncurr,lines, card, pot1, pot2=None):
            params = {}
            end = False
            n = 0 
            values = False
            card = card.replace("-", "_")
            card = f"{card}_{pot1}" if card == "poten" else f"{card}_{pot1}_{pot2}"

            while values == False:
                l = lines[n].split()[0]
                if "values" in str(l).lower():
                    values = True
                n+=1
            while end == False:
                l = lines[n].split()
                if "end" in str(l[0]).lower():
                    end = True
                    continue
                params[f"{card}_{str(l[0])}"]=[float(l[1]),n+ncurr]
                n+=1
            return params, n

        cards = [
            "poten",
            "spin-orbit-x",
            "lx",
            "bob-rot",
            "spin-rot",
            "lambda-p2q",
            "lambda-q"
            ]
        
        if extra_cards is not None:
            extra_cards = [extra_cards] if len(extra_cards) == 1 else extra_cards
            for card in extra_cards:
                if card not in cards:
                    cards.append(card)
        
        with open(input_file, "r") as file:
            lines = file.readlines()

        n = 0
        done = False
        params_dict = {}
        while done == False:
            if len(lines[n].split()) > 1:
                l    = lines[n].split()
                card = str(l[0].lower())
            else:
                n+=1
                continue
            if card in cards:
                pot1 = str(l[1])
                pot2 = None if len(l) == 2 else str(l[2]) 
                if card == "poten":
                    current_params_dict, n_add = read_in_values(n,lines[n:], card, pot1)
                    params_dict.update(current_params_dict)
                else:
                    current_params_dict, n_add = read_in_values(n,lines[n:], card, pot1, pot2)
                    params_dict.update(current_params_dict)
                n+=n_add
            n+=1

            if n >= len(lines)-1:
                done = True    
        
        self.params_dict = params_dict
        self.varied  = False
        self.bounded = False

    def set_varying_parameters(self, vary):
        for key in self.params_dict.keys():
            if self.varied == False:
                self.params_dict[key].insert(2,vary.get(key,False))
            else:
                self.params_dict[key][2] = vary.get(key,False)
        
        self.varied = True
        return self.params_dict

    def set_parameter_bounds(self, bounds):
        for key in self.params_dict.keys():
            if self.bounded == False:
                self.params_dict[key].extend(bounds.get(key,[-np.inf,np.inf]))
            else:
                self.params_dict[key][3:] = bounds.get(key,[-np.inf,np.inf])
        self.bounded = True
        return self.params_dict

    def parameter_table(self):
        names = ["value","line"]
        output_names = ["curve","parameter","value","line"]

        if self.varied == True:
            names.insert(2,"vary")
            output_names.insert(4, "vary")
        if self.bounded == True:
            names.extend(["lbound","ubound"])
            output_names.insert(5,"lbound")
            output_names.insert(6,"ubound")
        df = pd.DataFrame.from_dict(self.params_dict, orient = "index", columns = names)
        df["Index"] = df.index
        df["parameter"] = df.apply(lambda x:x["Index"].split("_")[-1], axis = 1)
        df["curve"]     = df.apply(lambda x:" ".join(x["Index"].split("_")[:-1]), axis = 1)
        df = df[output_names].reset_index(drop = True)
        return df

    def fit_input_file(self,marvel,duo_command,states_file_name, method,**kwargs):

        Q_nums = kwargs.get("Q_nums", ["J","tau","Manifold","v","Lambda", "Sigma","Omega"])
        Energy_handle = kwargs.get("Energy_hanlde", "E")

        index_dict = {key: val[1] for key, val in self.params_dict.items()}
        fcn_arguments = (index_dict,self.input_file, duo_command, states_file_name, marvel, Q_nums, Energy_handle)

        log_name = f"{self.input_file}.{method}.log"

        def get_unique_log_filename(base_filename):
            if not os.path.exists(base_filename):
                return base_filename

            name, extension = os.path.splitext(base_filename)
            counter = 1

            while os.path.exists(f"{name}{counter}{extension}"):
                counter += 1

            return f"{name}{counter}{extension}"
            
        log_name = get_unique_log_filename(log_name)

        def full_Iteration(params_dict,index_dict,input_file,duo_command,states_file_name, marvel, Q_nums, Energy_handle):
            try:
                iter = run_Iteration(input_file,duo_command,states_file_name,params_dict,index_dict)
                rms, merged = get_residuals(iter, marvel, Q_nums, Energy_handle)
                params_dict_formatted = {key:val.value for key,val in params_dict.items() if val.vary == True}
                print(f"RMS = {rms}")

                with open(log_name,"a") as file:
                    file.write("\n")
                    file.write(f"RMS = {rms}\n")
                    file.write(f"Parameters:\n")
                    for key,val in params_dict_formatted.items():
                        file.write(f"\t{key} \t {val}\n")

                return rms
            except:
                rms = np.nan
                return rms
            
        fitting_parameters = Parameters()
        
        for key,value in self.params_dict.items():
            fitting_parameters.add(key, value = value[0], vary=value[2], min = value[3], max = value[4])
        
        minner = Minimizer(full_Iteration, fitting_parameters, fcn_args = fcn_arguments, **kwargs)

        result = minner.minimize(method=method)
        return result


