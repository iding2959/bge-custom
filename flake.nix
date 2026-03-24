{
  description = "BGE-M3 embedding service development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonVersion = "3.11";
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            uv
            git
          ];

          shellHook = ''
            export UV_PYTHON_PREFERENCE=only-managed
            export UV_PYTHON=${pythonVersion}

            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment with Python ${pythonVersion}..."
              uv venv --python ${pythonVersion}
            fi

            source .venv/bin/activate

            if [ -f "pyproject.toml" ]; then
              echo "Syncing dependencies..."
              uv sync --all-extras
            fi

            echo "Python environment ready: $(python --version)"
          '';
        };
      });
}
