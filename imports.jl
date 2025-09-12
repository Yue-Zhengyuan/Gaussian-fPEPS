using Printf
using Glob
using ArgParse
using Parameters
using TensorKit
using PEPSKit
using TensorKitIO
using GaussianfPEPS
using HDF5
import TensorKitTensors.HubbardOperators as hub
import TensorKitTensors.TJOperators as tJ
using LRUCache
# set TensorKit cache size
resize!(TensorKit.GLOBAL_FSTRANSPOSE_CACHE; maxsize = 400)
resize!(TensorKit.GLOBAL_FSBRAID_CACHE; maxsize = 400)
resize!(TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE; maxsize = 400)
resize!(TensorKit.GLOBAL_TREEBRAIDER_CACHE; maxsize = 400)
resize!(TensorKit.GLOBAL_TREEPERMUTER_CACHE; maxsize = 400)
resize!(TensorKit.GLOBAL_TREETRANSPOSER_CACHE; maxsize = 400)
using Logging

# Define a custom logger subtype
struct SimpleLogger <: AbstractLogger
    min_level::LogLevel  # Minimum log level to handle
end

# Determine the minimum log level the logger handles
Logging.min_enabled_level(logger::SimpleLogger) = logger.min_level

# Check if a given log level is enabled
function Logging.shouldlog(logger::SimpleLogger, level::LogLevel, _module, _group, _id)
    return level >= logger.min_level
end

# Handle log messages
function Logging.handle_message(
        logger::SimpleLogger,
        level::LogLevel,
        message,
        _module,
        group,
        id,
        file,
        line;
        kwargs...,
    )
    # Print the message to stdout with metadata
    println(stderr, "[", level, "] ", message)
    flush(stderr)
    return flush(stdout)
end

# Example usage
logger = SimpleLogger(Logging.Info)  # Create a logger with a minimum level of Info
global_logger(logger)                # Set it as the global logger
