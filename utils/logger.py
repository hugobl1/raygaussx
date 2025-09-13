import subprocess

def log_nvidia_smi(logger):
    """Exécute nvidia-smi, omet la première ligne, et logge le reste en un seul appel."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )
        # Sépare en lignes et saute la première (l'en-tête)
        lines = result.stdout.strip().splitlines()[1:]
        if not lines:
            return

        # Recompose le bloc à logger
        output = "\n".join(lines)
        logger.info("NVIDIA-SMI GPU STATUS:\n%s", output)


    except subprocess.CalledProcessError as e:
        logger.error(
            "Erreur lors de l'exécution de nvidia-smi (%d): %s",
            e.returncode, e.stderr.strip()
        )
        
def log_phase(logger, title):
    offset=len(title)
    sep = f"{'─' * (60 - offset)} {title} {'─' * (60)}"
    logger.info(sep)