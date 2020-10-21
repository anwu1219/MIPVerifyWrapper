using Pkg

# Interface:
# RunMIPVerifySatisfiability environment_path property.txt network.nnet output_file strategy timeout_per_node
# For parsing the arguments to the file
using ArgParse
arg_settings = ArgParseSettings()
@add_arg_table! arg_settings begin
    "--environment_path"
        help = "Base path to your files. We will activate this package environment"
        arg_type = String
        default = "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper"
    "--property_file"
        help = "Property file name"
        arg_type = String
        required = true
    "--network_file"
        help = "Network file name"
        arg_type = String
        required = true
    "--output_file"
        help = "Output file name"
        arg_type = String
        required = true
    "--tightening"
        help = "Tightening strategy - mip, lp or ia"
        arg_type = String
        default = "mip"
    "--timeout_per_node"
        help = "Timeout in seconds per node"
        arg_type = Float64
        default = 20.0
    "--num_threads"
        help = "Number of threads"
        arg_type = Int64
        default = 1
    "--set_obj"
        help = "Set an objective for satisfiability check"
	arg_type = Bool
	default = false
end

# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
println(ARGS)
environment_path = parsed_args["environment_path"]
property_file_name = parsed_args["property_file"]
network_file_name = parsed_args["network_file"]
tightening = parsed_args["tightening"]
timeout_per_node = parsed_args["timeout_per_node"]
output_file_name = parsed_args["output_file"]
num_threads = parsed_args["num_threads"]
set_obj = parsed_args["set_obj"]

Pkg.activate(environment_path)

using JuMP
using Gurobi
using Interpolations
using MathProgBase
using CPUTime
using MathOptInterface

include(string(environment_path, "src/MIPVerify.jl"))
MIPVerify.set_log_level!("notice")

# Include util functions and classes to define our network
using Parameters # For a cleaner interface when creating models with named parameters
include("./activation.jl")
include("./network.jl")
include("./util.jl")


# Decide on your bound tightening strategy
if tightening == "lp"
   strategy = MIPVerify.lp
elseif tightening == "mip"
   strategy = MIPVerify.mip
elseif tightening == "ia"
    strategy = MIPVerify.interval_arithmetic
else
    println("Didn't recognize the tightening strategy")
    @assert false
end


# Run simple problem to avoid Sherlock startup time being counted
start_time = time()
println("Starting simple example")
simple_nnet = read_nnet(string(environment_path, "solverWrapper/Networks/small_nnet.nnet"))
simple_mipverify_network = network_to_mipverify_network(simple_nnet)
simple_property_lines = readlines(string(environment_path, "solverWrapper/Properties/small_nnet_property.txt"))
simple_lower_bounds, simple_upper_bounds = bounds_from_property_file(simple_property_lines, 1, simple_nnet.lower_bounds, simple_nnet.upper_bounds)

temp_p = get_model(
      (1,),
      simple_mipverify_network,
      Gurobi.Optimizer,
      lower_bounds=simple_lower_bounds,
      upper_bounds=simple_upper_bounds,
      )

add_output_constraints_from_property_file!(temp_p.model, temp_p.output_variable, simple_property_lines)
set_optimizer(temp_p.model, Gurobi.Optimizer)
optimize!(temp_p.model)
println("Finished simple solve in: ", time() - start_time)


# Read in the network and convert to a MIPVerify network
network = read_nnet(network_file_name)
mipverify_network = network_to_mipverify_network(network, "test", strategy)
num_inputs = size(network.layers[1].weights, 2)

# Read in the input upper and lower bounds from the property file
property_lines = readlines(property_file_name)
lower_bounds, upper_bounds = bounds_from_property_file(property_lines, num_inputs, network.lower_bounds, network.upper_bounds)

# Propagate bounds using MIPVerify
# Start timing
CPUtic()


p1 = get_model(
      (num_inputs,),
      mipverify_network,
      Gurobi.Optimizer,
      lower_bounds=lower_bounds,
      upper_bounds=upper_bounds,
      tightening_options=Dict("OutputFlag" => 0, "TimeLimit" => timeout_per_node, "Threads" => num_threads),
      tightening_algorithm=strategy
      )

preprocessing_time = CPUtoc()
println("Preprocessing took: ", preprocessing_time)

CPUtic()
# Add output constraints
add_output_constraints_from_property_file!(p1.model, p1.output_variable, property_lines)
	    
set_optimizer(p1.model, Gurobi.Optimizer)
set_optimizer_attributes(p1.model, Dict("Threads" => num_threads)...)

if set_obj
   #@objective(p1.model, Min, max(abs(p1.input_variable .- centroid) ./ (upper_bounds .- lower_bounds)))
   centroid = ( lower_bounds + upper_bounds ) / 2
   @objective(p1.model, Min, MIPVerify.get_norm(Inf, (p1.input_variable - centroid) / (upper_bounds - lower_bounds)))
else
    @objective(p1.model, Max, 0)
 end


# Solve the feasibility problem
optimize!(p1.model)
solve_time = CPUtoc()
status = JuMP.termination_status(p1.model)

println(status)

if (status == MathOptInterface.INFEASIBLE)
      println("Infeasible, UNSAT")
elseif (status == MathOptInterface.INFEASIBLE_OR_UNBOUNDED)
      println("Infeasible or unbounded, UNSAT")
elseif (status == MathOptInterface.OPTIMAL)
      println("Optimal, SAT")
elseif (status == MathOptInterface.OBJECTIVE_LIMIT)
      println("SAT")
else
      println("Unknown!")
end

println("Preprocessing time: ", preprocessing_time)
println("Solve time: ", solve_time)

# Write to the output file the status, objective value, and elapsed time
output_file = string(output_file_name) # add on the .csv
open(output_file, "w") do f
    # Writeout our results
    if ( status == MathOptInterface.INFEASIBLE || status == MathOptInterface.INFEASIBLE_OR_UNBOUNDED )
       	 write(f,
       	 "unsat ", 
       	 string(preprocessing_time + solve_time), " ",
       	  "\n")
    elseif ( status == MathOptInterface.OPTIMAL || status == MathOptInterface.OBJECTIVE_LIMIT )
       	 write(f,
       	 "sat ", 
       	 string(preprocessing_time + solve_time), " ",
       	  "\n")
    else
       	 write(f,
       	 "UNKNOWN ", 
       	 string(preprocessing_time + solve_time), " ",
       	  "\n")
    end
   close(f)
end
