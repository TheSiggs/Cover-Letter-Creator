{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pkgs.poetry
            pkgs.gcc
            pkgs.libffi
            pkgs.cmake
          ];
          shellHook = ''
            echo "Activating Poetry environment..."
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
            if [ ! -d ".venv" ]; then
              poetry install
            fi
            source .venv/bin/activate
            exec zsh
          '';
        };
      });
}

