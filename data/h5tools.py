# h5tools.py

import tables as tb
import pandas as pd
import numpy as np
import os
from typing import Union, Any, Iterable, Optional
from collections.abc import Mapping, Sequence
import warnings
import torch
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn, rank_zero_info
import time


class TableSequence(Mapping, Sequence):
    """
    A class that allows for indexing and slicing of HDF5 tables, without having 
    to open the table every time, so that the table behaves as a Sequence.

    Parameters:
    ----------
    path (str): 
        The path to the HDF5 file containing the table.
    table_name (str): 
        The name of the table to index.

    Raises:
    -------
    OSError: 
        If the path does not point to a valid HDF5 file.
    ValueError: 
        If the table does not exist in the file.
    """
    def __init__(
            self, 
            path : str, 
            root_uep : str, 
            table_name : str,
            read_only : bool = False,
            ) -> None:
        # Verify that the path points to a valid HDF5 file
        if os.path.isdir(path):
            raise OSError('The path points to a directory, not a file.')
        filetype = os.path.splitext(path)[-1]
        if filetype != '.h5':
            raise OSError(f'The path points to a {filetype} file, not an .h5 one.')
        self.path = path
        self.root_uep = root_uep
        self.read_only = read_only

        # Verify that the table indeed belongs to the file
        with tb.open_file(path, 'r', root_uep=root_uep) as f:
            if not table_name in f.root:
                raise ValueError(f'The table {table_name} does not exist in {path}.')
        self.table_name = table_name


    def __getitem__(self, idx : Union[int, slice, str]) -> np.ndarray:
        """
        Returns the row(s) or column(s) of the table specified by the given 
        index. If the index is an integer or slice, the row(s) are returned. If 
        the index is a string, the column(s) are returned.

        Parameters:
        ----------
        idx (int | slice | str): 
            The index of the row(s) or column(s) to return.

        Returns:
        -------
        np.ndarray: 
            The row(s) or column(s) of the table specified by the given index.
        """
        if isinstance(idx, (int, slice)):
            with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
                return f.root[self.table_name][idx]
        else:
            with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
                return f.root[self.table_name].col(idx)
        

    def __len__(self) -> int:
        """
        Get the number of rows in the table.

        Returns:
        -------
        int: 
            The number of rows in the table.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return len(f.root[self.table_name])
        
    
    def __contains__(self, col: str) -> bool:
        """
        Check if the table contains the given column.

        Parameters:
        ----------
        col (str): 
            The column to check for.

        Returns:
        -------
        bool: 
            True if the table contains the given column, False otherwise.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return col in f.root[self.table_name].colnames
      

    def __repr__(self) -> str:
        """
        Produces a string representation of the table.

        Returns:
        -------
        str: 
            A string representation of the table.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return f.root[self.table_name].__repr__()
        

    def read(self) -> np.ndarray:
        """
        Read the entire table into memory.

        Returns:
        -------
        np.ndarray: 
            The entire table as a structured numpy array.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return f.root[self.table_name].read()
    
        
    def append(self, data : np.ndarray) -> None:
        """
        Append the given data to the table. There are currently no checks as to
        whether the data is compatible with the table, but it will not throw
        an error if it is not.

        Parameters:
        ----------
        data (np.ndarray): 
            The data to append to the table.
        """
        if self.read_only:
            raise ValueError('The file is read-only.')
        with tb.open_file(self.path, 'a', root_uep=self.root_uep) as f:
            f.root[self.table_name].append(data)


    def size_in_memory(self) -> int:
        """
        Get the size of the file in memory, in bytes.

        Returns:
        -------
        int: 
            The size of the file in memory.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return f.root[self.table_name].get_filesize()
        
    
    def to_csv(self, path : str) -> None:
        """
        Write the table to a CSV file. Unfortunately for now, this involves 
        reading the entire table into memory.

        Parameters:
        ----------
        path (str): 
            The path to write the CSV file to.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            pd.DataFrame(
                f.root[self.table_name].read()
                ).to_csv(path, index=False)





class HDF5File(Mapping):
    """
    A class that enables working with HDF5 files, especially ones that contain
    groups of tables. This class is tuned
    to work in distributed settings as well.

    Parameters:
    ----------
    path (str): 
        The path to the HDF5 file containing the tables.
    root_uep (str, optional):
        The root user entry point. Setting this means that only tables and 
         groups within this group are visible to the class instance. Defaults to
        '/'.
    create_file_if_not_exists (bool, optional):
        Whether to create a blank file at the specified path if one does not 
        already exist. Defaults to True. Setting this to false is helpful
        for debugging purposes.
    verbose (bool, optional):
        Whether to print warnings and other messages. Defaults to False.

    Raises:
    -------
    OSError: 
        If the path does not point to a valid HDF5 file.
    """
    def __init__(
            self, 
            path : str, 
            root_uep : str = '/', 
            verbose : bool = False,
            read_only : bool = False,
            ) -> None:
        
        # store the arguments
        self.path = path
        self.root_uep = root_uep
        self.read_only = read_only
        self.verbose = verbose

        # create the file if it does not exist
        if not os.path.exists(self.path):
            self.create_blank_file() 

        # verify that the file is valid
        self.verify_file()


    @rank_zero_only
    def create_blank_file(self) -> None:
        if self.read_only:
            raise OSError(
                f'The file at the path {self.path} does not exist, and the file is read-only, so it cannot be created.'
                )
        if self.verbose:
            rank_zero_warn(
                'There is no file at the specified path. A blank file has been created for you.'
                )
        with tb.open_file(self.path, 'w') as f: 
            pass

    
    @rank_zero_only
    def create_blank_group(self, group_name : str) -> None:
        if self.read_only:
            raise OSError(
                f'The file is read-only, so the blank group {group_name} cannot be created.'
                )
        with tb.open_file(self.path, 'a') as f:
            f.create_group(f.root, group_name)


    @rank_zero_only
    def verify_file(self) -> None:
        """
        Verify that the file is valid, by opening it and closing it. Then, 
        check whether the root_uep exists. If it does not, create it.
        """
        with tb.open_file(self.path, 'r') as f:
            root_uep_exists = self.root_uep == '/' or self.root_uep in f.root
        if not root_uep_exists:
            if self.verbose:
                rank_zero_warn(
                    'The root_uep does not exist. A blank group has been created for you.'
                    )
            self.create_blank_group(self.root_uep)      
                

    def __len__(self) -> int:
        """
        Get the number of nodes in the root directory of the file.

        Returns:
        -------
        int: 
            The number of nodes in the root directory.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return f.root._g_getnchildren()   
    
    
    def walk_nodes(self, classname : Optional[str] = None) -> Iterable:
        """
        Iterate over all tables in the file, regardless of their depth.

        Returns:
        -------
        Iterable: 
            An iterator over all the tables in the file.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            for x in f.walk_nodes(f.root, classname=classname):
                yield x


    def num_rows(self) -> int:
        """
        Get the number of rows in all tables in the file.

        Returns:
        -------
        int:
            The number of rows in all tables in the file.
        """
        return sum([len(table) for table in self.walk_nodes('Table')])       
    

    def __getitem__(self, key : str) -> TableSequence:
        """
        Get the table or group with the given name.

        Parameters:
        ----------
        key (str): 
            The name of the table or group to get.

        Returns:
        -------
        TableSequence | GroupedTable | np.ndarray: 
            If the key points to a table, a TableSequence object containing the
            table. If the key points to a group, a GroupedTable object
            containing the group. Else, if the key points to an Array,
            the Array is returned in memory as a numpy array.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            if isinstance(f.root[key], tb.Table):
                return TableSequence(self.path, self.root_uep, key)
            elif isinstance(f.root[key], tb.Array):
                return f.root[key].read()
        return self.__class__(
            self.path, 
            os.path.join(self.root_uep, key)
            )
    

    def __contains__(self, key : str) -> bool:
        """
        Check if the file contains a table or group with the given name.

        Parameters:
        ----------
        key (str): 
            The name of the table or group to check for.

        Returns:
        -------
        bool: 
            True if the file contains a table or group with the given name, 
            False otherwise.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return key in f.root
        
    
    def __iter__(self) -> Iterable:
        """
        Iterate over the groups and tables in the file.

        Returns:
        -------
        Iterable: 
            An iterator over the groups and tables in the file.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            for group in f.root:
                yield group._v_name


    @rank_zero_only
    def create_blank_table(
            self, 
            table_name : str, 
            description : dict,
            complevel : int = 0,
            expectedrows : int = 0,
            ) -> None:
        """
        Create a length-zero table of the given name and description in the 
        current HDF5 file.

        Parameters:
        ----------
        table_name (str): 
            The name of the table to create.
        description (dict): 
            A dictionary describing the structure of the table.
        complevel (int, optional): 
            The compression level to use when writing the table. Defaults to
            0, which means no compression. 
        expectedrows (int, optional): 
            The expected number of rows in the table. Defaults to 0, but 
            specifying this value more accurately can improve performance.
        """
        if self.read_only:
            raise ValueError('The file is read-only.')

        if table_name in self:
            raise ValueError(f'The table {table_name} already exists in {self.path}.')
        
        FILTERS = tb.Filters(complib='blosc', complevel=complevel)
        with tb.open_file(
            filename = self.path, 
            mode = 'a',
            filters = FILTERS, 
            root_uep = self.root_uep
            ) as f:            
            f.create_table(
                where = f.root, 
                name = table_name, 
                description = description, 
                expectedrows = expectedrows,
                )
            
    
    @rank_zero_only
    def table_from_struct(
            self, 
            table_name : str, 
            data : np.ndarray,
            complevel : int = 0,
            ) -> None:
        """
        Create a new table of the given name and description in the current HDF5
        file, and populate it with the given data.

        Parameters:
        ----------
        table_name (str): 
            The name of the table to create.
        data (np.ndarray): 
            The data to populate the table with, in the form of a structured
            numpy array.
        complevel (int, optional):
            The compression level to use when writing the table. Defaults to
            0, which means no compression.

        Raises:
        -------
        ValueError: 
            If a table with the given name already exists in the file.
        """
        if self.read_only:
            raise ValueError('The file is read-only.')
        if table_name in self:
            raise ValueError(f'The table {table_name} already exists in {self.path}.')
        if table_name.startswith(self.root_uep):
            raise ValueError(f'The table name {table_name} cannot start with the root_uep {self.root_uep}.')
        
        # Define the table structure using the data's dtypes
        description = tb.descr_from_dtype(data.dtype)

        # Create a new HDF5 file and table
        FILTERS = tb.Filters(complib='blosc', complevel=complevel)
        with tb.open_file(
            filename = self.path, 
            mode = 'a',
            filters = FILTERS,
            root_uep = self.root_uep
            ) as f:
            f.create_table(
                where = f.root, 
                name = table_name, 
                description = description[0], 
                expectedrows = len(data),
                )
            
            # Append the data to the table
            f.root[table_name].append(data)


    @rank_zero_only
    def table_from_csv(
            self,
            csv_file : str, 
            table_name : str, 
            read_cols : list,
            complevel : int = 0, 
            transforms : Iterable[callable] = [],
            ) -> None:
        """
        Create a new table of the given name and description in the current HDF5
        file, and populate it with data from the given CSV file.

        Parameters:
        ----------
        csv_file (str): 
            The path to the CSV file to read from.
        table_name (str): 
            The name of the table to create.
        read_cols (list): 
            A list of column names to read from the CSV file.
        complevel (int, optional): 
            The compression level to use when writing the table. Defaults to
            0, which means no compression.
        transforms (Iterable[callable], optional):
            An iterable of functions to apply to the data before writing it to 
            the table. Defaults to an empty list.

        Raises:
        -------
        ValueError: 
            If a table with the given name already exists in the file.
        """
        if self.read_only:
            raise ValueError('The file is read-only.')
        if table_name in self:
            raise ValueError(f'The table {table_name} already exists in {self.path}.')
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file, usecols=read_cols)
        
        # Convert the DataFrame to a structured numpy array
        data = df.to_records(index=False)

        # Apply any transforms to the data, in the order they were given
        for transform in transforms:
            data = transform(data)

        # Create a new table from the transformed data
        self.table_from_struct(table_name, data, complevel)


    def to_csvs(self, path : str) -> None:
        """
        Write each table in the file to a CSV file.

        Parameters:
        ----------
        path (str): 
            The directory to write the CSV files to.

        Raises:
        -------
        OSError: 
            If the path does exist, or does not point to a directory.
        """
        if not os.path.exists(path):
            raise OSError('The path does not exist.')
        if not os.path.isdir(path):
            raise OSError('The path must point to a directory.')

        for table_name in self:
            self[table_name].to_csv(os.path.join(path, table_name + '.csv'))


    def __repr__(self) -> str:
        """
        Produces a string representation of the file.

        Returns:
        -------
        str: 
            A string representation of the file.
        """
        tables = ''.join([f'{key}, of length {len(self[key])}\n    ' for key in self])
        return 'A GroupedTables object containing the following tables:\n    ' + tables
    

    def delete_node(self, node_name : str) -> None:
        """
        Delete the node with the given name.

        Parameters:
        ----------
        node_name (str): 
            The name of the node to delete.
        """
        if self.read_only:
            raise ValueError('The file is read-only.')
        if not node_name in self:
            raise ValueError(f'The node {node_name} does not exist in {self.path}.')
        with tb.open_file(self.path, 'a', root_uep=self.root_uep) as f:
            f.remove_node(f.root, node_name, recursive=True)

    
    def read(self):
        """
        Read the entire file into memory.

        Returns:
        -------
        np.ndarray: 
            The entire file as a structured numpy array.
        """
        with tb.open_file(self.path, 'r', root_uep=self.root_uep) as f:
            return f.root.read()
    




from torch.utils.data import IterableDataset
import math
class MappingIterableDataset(IterableDataset):
    def __init__(
            self, 
            mapping : Mapping[Any, Sequence], 
            batch_size : int, 
            rank : int = 0,
            world_size : int = 1,
            ) -> None:
        self.mapping = mapping
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size


    def __iter__(self) -> Iterable:
        while True:
            for table in self.mapping.values():
                slices = [
                    slice(i, i+self.batch_size) 
                    for i in range(0, len(table), self.batch_size)
                    ]
                slices = slices[self.rank::self.world_size]
                for s in slices:
                    yield table[s]

    
    def __len__(self) -> int:
        return math.ceil(self.mapping.num_rows() / self.batch_size)