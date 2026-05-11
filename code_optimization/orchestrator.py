"""Optimization Orchestrator for the program optimization system.

This module implements the main orchestration logic that coordinates the
optimization process across multiple iterations. It manages the Swarm, tracks
results, and extracts insights from agent interactions.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from .agents import create_analysis_agent, create_researcher_agent, create_supervisor_agent
from .core.streaming_logger import StreamingConversationLogger
from strands.multiagent import Swarm

logger = logging.getLogger(__name__)


class OptimizationOrchestrator:
    """Orchestrates a single program optimization using a Swarm.

    The orchestrator manages the optimization workflow:
    1. Initialize components (Swarm, ProgramManager, Evaluator)
    2. Execute the Swarm to get proposed changes
    3. Extract rationale and findings from agent messages
    4. Create modified program version
    5. Evaluate the modified program
    6. Return results with program and evaluation
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluator_path: str,
        output_dir: str,
        phase_1_iterations: int,
        phase_2_iterations: int,
    ):
        """Initialize the Optimization Orchestrator.

        Args:
            initial_program_path: Path to the initial program with EVOLVE-BLOCKs
            evaluator_path: Path to the evaluator module
            output_dir: Directory for output files (default: creates timestamped dir)

        Raises:
            ValueError: If initial program or evaluator are invalid
            FileNotFoundError: If required files don't exist
        """
        self.phase_1_iterations = phase_1_iterations
        self.phase_2_iterations = phase_2_iterations
        self.initial_program_path = initial_program_path
        self.evaluator_path = evaluator_path
        self.output_dir = Path(output_dir)

        logger.info(
            "Initialized OptimizationOrchestrator: program=%s, evaluator=%s, output=%s",
            initial_program_path,
            evaluator_path,
            output_dir,
        )

    def _setup_output_directory(self, initial_program_name: str) -> None:
        """Create the output directory structure."""
        logger.debug("Setting up output directory structure at %s", self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "programs").mkdir(exist_ok=True)
        (self.output_dir / "conversations").mkdir(exist_ok=True)

        # Copy initial program to programs folder
        import shutil

        dest_path = self.output_dir / "programs" / initial_program_name
        shutil.copy2(self.initial_program_path, dest_path)
        logger.info("Copied initial program to %s", dest_path)

    def run(self):
        """Execute a single optimization using the Swarm.

        Returns:
            OptimizationResults containing the optimized program and evaluation
        """
        logger.info("Starting optimization process")
        start_time = time.time()

        try:
            logger.info("Running optimization with program: %s", self.initial_program_path)

            initial_program_name = f"v00_{Path(self.initial_program_path).name}"
            self._setup_output_directory(initial_program_name)
            analysis_program_path = (self.output_dir / "programs" / initial_program_name).resolve()

            # Set up streaming logger for real-time conversation logging
            log_path = self.output_dir / "conversations" / "optimization.txt"
            streaming_logger = StreamingConversationLogger(log_path)

            # Run Phase 1 first: analysis-only pass that improves theoretical bounds
            self._run_phase_1(streaming_logger, analysis_program_path)

            phase_2_program_path = self._get_latest_program_path()
            logger.info("Phase 2 will start from %s", phase_2_program_path)

            # Log the phase switch to the streaming conversation log
            try:
                streaming_logger._append_to_log("\n *** Switching to Phase 2 ***\n")
            except Exception:
                logger.exception("Failed to write phase switch to streaming log")

            swarm = self._create_swarm()

            # Set the callback handler on all agents in the swarm
            for node in swarm.nodes.values():
                if hasattr(node, "executor") and hasattr(
                    node.executor, "callback_handler"
                ):
                    node.executor.callback_handler = streaming_logger

            # Execute the Swarm with streaming logger
            context = self._build_optimization_context(phase_2_program_path)
            logger.info("Executing Swarm (Researcher <-> Supervisor)")
            swarm_result = swarm(context)
            logger.info("Swarm execution completed")

            # Finalize the streaming log
            streaming_logger.finalize()

        except Exception as e:
            logger.exception("Optimization failed: %s", str(e))
            raise

    def _run_phase_1(
        self,
        streaming_logger: StreamingConversationLogger,
        analysis_program_path: Path,
    ) -> None:
        """Run the analysis agent before the researcher/supervisor swarm."""
        logger.info("Executing Phase 1 (Analysis) with %d iterations", self.phase_1_iterations)

        analysis_agent = create_analysis_agent()
        analysis_swarm = Swarm(
            nodes=[analysis_agent],
            entry_point=analysis_agent,
            max_handoffs=self.phase_1_iterations,
            max_iterations=self.phase_1_iterations,
            node_timeout=3600,
        )

        for node in analysis_swarm.nodes.values():
            if hasattr(node, "executor") and hasattr(node.executor, "callback_handler"):
                node.executor.callback_handler = streaming_logger

        context = self._build_phase_1_context(analysis_program_path)
        logger.info("Executing Analysis Swarm")
        analysis_swarm(context)
        logger.info("Phase 1 completed")

    def _build_phase_1_context(self, analysis_program_path: Path) -> str:
        """Build the prompt for the analysis-only phase."""
        analysis_program_path = analysis_program_path.resolve()
        context_parts = [
            f"Program to Analyze: {analysis_program_path}",
            "",
            "TASK:",
            (
                "Analyze the current program and iteratively improve its asymptotic bounds. "
                "For each iteration, identify the current complexity, propose a better algorithm, "
                "and update the EVOLVE-BLOCK implementation to reflect the theoretical improvement. "
                "Write each new version using the filename format vXX_<name>.py "
                "(for example: v01_initial_program.py, v02_initial_program.py). "
                f"Use the exact path above when reading the program, and write new versions into {self.output_dir}/programs."
            ),
        ]

        # Explicitly instruct the Analysis Agent to limit the number of
        # iterations/versions it produces. This prevents the agent from
        # generating more than the CLI-requested number of Phase 1 iterations
        # when it internally loops or performs multiple replacements.
        context_parts.append("")
        context_parts.append(f"MAX_PHASE_1_ITERATIONS: {self.phase_1_iterations}")
        context_parts.append(
            f"Important: stop after producing exactly {self.phase_1_iterations} new program version(s)."
        )

        return "\n".join(context_parts)

    @staticmethod
    def _extract_version_number(program_path: Path) -> Optional[int]:
        """Extract version number from filenames like v01_name.py or v12_name.py."""
        match = re.match(r"^v(\d+)_.*\.py$", program_path.name)
        if not match:
            return None
        return int(match.group(1))

    def _get_latest_program_path(self) -> Path:
        """Return the most recent program artifact produced in the programs folder."""
        programs_dir = self.output_dir / "programs"
        program_files = sorted(programs_dir.glob("*.py"))

        if not program_files:
            return (programs_dir / f"v00_{Path(self.initial_program_path).name}").resolve()

        # Prefer explicit versioned files with vXX_ naming.
        versioned_files = []
        for program_file in program_files:
            version_num = self._extract_version_number(program_file)
            if version_num is not None:
                versioned_files.append((version_num, program_file.name, program_file))

        if versioned_files:
            # Highest version wins; filename is a stable tie-breaker.
            _, _, latest_file = max(versioned_files, key=lambda x: (x[0], x[1]))
            return latest_file.resolve()

        # Fallback: if no file matches vXX_ pattern, use newest by mtime.
        logger.warning(
            "No versioned program files matching vXX_<name>.py were found in %s; "
            "falling back to newest .py artifact by modification time.",
            programs_dir,
        )
        latest_by_mtime = max(program_files, key=lambda p: p.stat().st_mtime)
        return latest_by_mtime.resolve()

    def _build_optimization_context(self, program_path: Path) -> str:
        """Build context for the Swarm.

        Args:
            program_path: Path to program to evolve

        Returns:
            Formatted context string for the Swarm
        """
        program_path = program_path.resolve()
        context_parts = []

        # Add program information
        context_parts.append(f"Program to Evolve: {program_path}")
        context_parts.append("")

        # Add task description
        context_parts.append("TASK:")
        context_parts.append(
            "Analyze the program and propose improvements to the EVOLVE-BLOCK sections. "
            "Follow the experimental methodology: form a hypothesis, implement changes, "
            "analyze results, and document findings."
            f"Store successive programs at {self.output_dir}/programs/vXX_<name>, where XX is the version number."
        )

        return "\n".join(context_parts)

    def _create_swarm(
        self,
        model_id: str = None,
    ) -> Swarm:
        """Create and configure the Swarm with Researcher and Supervisor agents.

        Args:
            model_id: Optional model ID to use. If None, uses MODEL_ID environment
                variable or defaults based on MODEL environment variable.
        """

        researcher = create_researcher_agent(model_id=model_id)
        supervisor = create_supervisor_agent(model_id=model_id)

        return Swarm(
            nodes=[researcher, supervisor],
            entry_point=researcher,
            max_handoffs=self.phase_2_iterations,
            max_iterations=self.phase_2_iterations,
            node_timeout=3600,
        )
