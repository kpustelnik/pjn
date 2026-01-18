{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  packages = with pkgs; [
    uv
    stdenv.cc.cc.lib
    zlib
  ];
}
