#! /usr/bin/env julia

#=

    Run a query from a line with the following format:

    optimizer_string path/to/network/network.nnet path/to/property/property.txt path/to/result/result_file.txt

    For example:
    MIPVerify_optimizer=Gurobi.Optimizer_threads=1_strategy=ia_preprocesstimeoutpernode=5 /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/ACASXu/ACASXU_experimental_v2a_1_1.nnet /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/OptimizationProperties/ACASXu/acas_property_optimization_1.txt /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/FGSM.ACASXU_experimental_v2a_1_1.acas_property_optimization_1.txt

    To run a quick test:
    module test
           ARGS = ["--environment_path", "/Users/castrong/Desktop/Research/MIPVerifyWrapper/", "--optimizer", "MIPVerify_optimizer=Gurobi.Optimizer_threads=1_strategy=mip_preprocesstimeoutpernode=2.0", "--network_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Networks/AutoTaxi_32Relus_200Epochs_OneOutput.nnet", "--property_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Properties/autotaxi_property_AutoTaxi_17201_transposed_0.01_max.txt", "--result_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/MIPVerify_optimizer=Gurobi.Optimizer_threads=1_strategy=mip_preprocesstimeoutpernode=2.0.AutoTaxi_32Relus_200Epochs_OneOutput.autotaxi_property_AutoTaxi_17201_transposed_0.01_max.txt"]
 	   	   include("/Users/castrong/Desktop/Research/MIPVerifyWrapper/RunMIPVerifyFromPropertyFile.jl")
	end

=#


using Pkg
using ArgParse
arg_settings = ArgParseSettings()
@add_arg_table! arg_settings begin
    "--environment_path"
        help = "Base path to your files. We will activate this package environment"
        arg_type = String
        default = "/Users/castrong/Desktop/Research/NeuralOptimization.jl"
	"--optimizer"
		help = "String describing the optimizer"
		arg_type = String
		default = "FGSM"
		required = true
    "--network_file"
        help = "Network file name"
        arg_type = String
        required = true
    "--property_file"
        help = "Property file name"
        arg_type = String
        required = true
	"--result_file"
		help = "Result file name"
		arg_type = String
		required = true
end

# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
println("Parsed args: ", parsed_args)
environment_path = parsed_args["environment_path"]
optimizer_string = parsed_args["optimizer"]
network_file = parsed_args["network_file"]
property_file = parsed_args["property_file"]
result_file = parsed_args["result_file"]

# Activate the environment and include NeuralOptiimization.jl
environment_path = joinpath(environment_path, "../MIPVerifyWrapper/") # SKETCHY ASSUMPTION MIPVerifyWrapper is in same directory as the given environment for NeuralOptimiization.jl
Pkg.activate(environment_path)

using LazySets
using Interpolations
using NPZ
using JuMP
using ConditionalJuMP
using LinearAlgebra
using MathProgBase
using CPUTime
using Memento
using AutoHashEquals
using DocStringExtensions
using ProgressMeter
using MAT
using GLPKMathProgInterface
# You can use your solver of choice
using Gurobi

# Include MIPVerify
include(string(environment_path, "MIPVerify.jl/src/MIPVerify.jl"))
MIPVerify.setloglevel!("info")

# Include util functions and classes to define our network
using Parameters # For a cleaner interface when creating models with named parameters
include(string(environment_path, "activation.jl"))
include(string(environment_path, "network.jl"))
include(string(environment_path, "util.jl"))


# If the result file is already there do nothing
if !isfile(result_file)

	# Run a simple problem to avoid startup time being counted
	start_time = time()
	println("Starting simple example")
	simple_nnet = read_nnet(joinpath(environment_path, "Networks/small_nnet.nnet"))
	simple_mipverify_network = network_to_mipverify_network(simple_nnet)
	temp_p = get_optimization_problem(
	      (1,),
	      simple_mipverify_network,
	      GurobiSolver(),
	      lower_bounds=[0.0],
	      upper_bounds=[1.0],
	      )
	@objective(temp_p.model, Max, temp_p.output_variable[1])
	solve(temp_p.model)
	println("Finished simple solve in: ", time() - start_time)


	# A problem needs a network, input set, objective and whether to maximize or minimize.
	# it also takes in the lower and upper bounds on the network input variables which describe
	# the domain of the network.

	# Parse the optimizer string
	backend_optimizer, threads, strategy, preprocess_timeout_per_node = parse_mipverify_string(optimizer_string)

	println("backend: ", backend_optimizer)
	println("threads: ", threads)
	println("Strategy: ", strategy)
	println("Preprocess timeout: ", preprocess_timeout_per_node)

	if (backend_optimizer == "Gurobi.Optimizer")
		main_solver = GurobiSolver(Threads=threads)
		tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=preprocess_timeout_per_node, Threads=threads)
	else
		@assert false "Only supporting Gurobi for now - just would need to fill in this else with a proper constructor"
	end

	println("network file: ", network_file)
	network = read_nnet(network_file)
	mipverify_network = network_to_mipverify_network(network, "label", strategy)
	num_inputs = size(network.layers[1].weights, 2)

	if occursin("mnist", network_file)
		println("MNIST network")
		lower = 0.0
		upper = 1.0
	elseif occursin("AutoTaxi", network_file)
		println("AutoTaxi network")
		lower = 0.0
		upper = 1.0
	elseif occursin("ACASXU", network_file)
		println("ACAS network")
		lower = -Inf
		upper = Inf
	else
		@assert false "Network category unrecognized"
	end

	input_set, objective, maximize_objective = read_property_file(property_file, num_inputs; lower=lower, upper=upper)

	CPUtic()
	opt_problem = get_optimization_problem(
	      (num_inputs,),
	      mipverify_network,
	      main_solver,
	      lower_bounds=low(input_set),
	      upper_bounds=high(input_set),
		  tightening_solver=tightening_solver,
		  summary_file_name=string(result_file, ".bounds.txt")
	      )

	preprocess_time = CPUtoc()
	CPUtic()

	# Appropriately setting the objective
	if maximize_objective
		@objective(opt_problem.model, Max, sum(opt_problem.output_variable[objective.variables] .* objective.coefficients))
	else
		@objective(opt_problem.model, Min, sum(opt_problem.output_variable[objective.variables] .* objective.coefficients))
	end
	# Perform the optimization then pull out the objective value and elapsed time
	solve(opt_problem.model)
	main_solve_time = CPUtoc()
	elapsed_time = preprocess_time + main_solve_time
	objective_value = getobjectivevalue(opt_problem.model)
	optimal_input = getvalue.(opt_problem.input_variable)
	optimal_output = compute_output(network, optimal_input)
	println("Optimal input: ", optimal_input)
	println("Optimal output: ", optimal_output)
	println("Output from mipverify network: ", mipverify_network(optimal_input))
	println("output from other network: ", compute_output(network, optimal_input))

	open(result_file, "w") do f
		# Writeout our results - for the optimal output we remove the brackets on the list
		print(f, "success", ",") # status
		print(f, string(objective_value), ",") # objective value
		print(f, string(elapsed_time), ",") # elapsed time
		println(f, string(preprocess_time)) # preprocessing time, end this line

		# We assume it was successful here
		println(f, string(optimal_input)[2:end-1]) # Write optimal input on its own line
		println(f, string(optimal_output)[2:end-1])# Write optimal output on its own line

		close(f)
	end

# The file already exists
else
	println("Result File: ", result_file)
	println("Result file already exists, skipping execution!")
end
