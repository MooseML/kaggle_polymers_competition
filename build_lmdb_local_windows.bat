@echo off
REM build_lmdb_local_windows.bat 


echo Building Polymer LMDBs Locally for Kaggle Upload


REM Create output directory 
if not exist "data\processed_chunks" mkdir "data\processed_chunks"


echo Building training LMDB...
python build_polymer_lmdb_fixed.py train


echo Building test LMDB...
python build_polymer_lmdb_fixed.py test


echo Cleaning up lock files...
REM Remove lock files on Windows
del "data\processed_chunks\*.lock" 2>nul
del "data\processed_chunks\lock.mdb" 2>nul


echo Created files:
dir "data\processed_chunks\"


echo Verifying LMDB integrity...
python -c "import lmdb, os; [(print(f'{name} LMDB: {lmdb.open(path, readonly=True, lock=False).begin().stat()[\"entries\"]:,} entries'), lmdb.open(path, readonly=True, lock=False).close()) for name, path in [('Train', 'data/processed_chunks/polymer_train3d_dist.lmdb'), ('Test', 'data/processed_chunks/polymer_test3d_dist.lmdb')] if os.path.exists(path)]"


echo Ready for upload to Kaggle!!!
echo Upload directory: data\processed_chunks\