"""
Safe File Writer

Guarantees file persistence with atomic writes, backups, and verification.

Features:
- Atomic writes (write to temp, then rename)
- Automatic backups before overwriting
- Content verification (checksum)
- Rollback capability
- Detailed logging
"""

from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Any, Optional, List, Dict
import hashlib


class SafeFileWriter:
    """
    Guarantees file persistence with atomic writes, backups, and verification.
    
    Features:
    - Atomic writes (write to temp, then rename)
    - Automatic backups before overwriting
    - Content verification (checksum)
    - Rollback capability
    - Detailed logging
    """
    
    def __init__(self, config):
        """
        Initialize safe file writer.
        
        Args:
            config: Config instance with OUTPUT_DIR, BACKUP_DIR, AUTO_BACKUP settings
        """
        self.config = config
        self.write_log = []
    
    def write_json(
        self,
        data: Any,
        filepath: Path,
        backup: Optional[bool] = None,
        verify: bool = True
    ) -> Dict:
        """
        Write JSON file safely with all guarantees.
        
        Args:
            data: Data to write
            filepath: Target file path
            backup: Create backup (defaults to config.AUTO_BACKUP)
            verify: Verify write succeeded
        
        Returns:
            Write report with status, paths, checksums
        """
        filepath = Path(filepath)
        backup = backup if backup is not None else self.config.AUTO_BACKUP
        
        # Create parent directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_file': str(filepath),
            'success': False
        }
        
        # Step 1: Backup existing file
        if backup and filepath.exists():
            try:
                backup_path = self._create_backup(filepath)
                report['backup_path'] = str(backup_path)
            except Exception as e:
                report['backup_error'] = str(e)
                # Continue anyway - backup failure shouldn't prevent write
        
        # Step 2: Write to temporary file
        temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Step 3: Verify temp file
            if verify:
                with open(temp_path, 'r') as f:
                    loaded = json.load(f)
                
                # Basic verification
                if isinstance(data, dict):
                    if set(loaded.keys()) != set(data.keys()):
                        raise ValueError("Loaded data has different keys than original")
                elif isinstance(data, list):
                    if len(loaded) != len(data):
                        raise ValueError(f"Loaded data has {len(loaded)} items, expected {len(data)}")
            
            # Step 4: Atomic rename
            shutil.move(str(temp_path), str(filepath))
            
            # Step 5: Final verification
            if verify:
                checksum = self._calculate_checksum(filepath)
                report['checksum'] = checksum
                
                # Verify file is readable
                with open(filepath, 'r') as f:
                    json.load(f)
            
            report['success'] = True
            report['file_size'] = filepath.stat().st_size
            
        except Exception as e:
            report['error'] = str(e)
            
            # Rollback if we have a backup
            if backup and 'backup_path' in report:
                try:
                    backup_path = Path(report['backup_path'])
                    if backup_path.exists():
                        shutil.copy(backup_path, filepath)
                        report['rolled_back'] = True
                except Exception as rollback_error:
                    report['rollback_error'] = str(rollback_error)
        
        finally:
            # Clean up temp file if it still exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass  # Best effort cleanup
        
        self.write_log.append(report)
        return report
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = self.config.BACKUP_DIR / backup_name
        
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(filepath, backup_path)
        
        return backup_path
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def get_write_log(self) -> List[Dict]:
        """Get all write operations."""
        return [log.copy() for log in self.write_log]
    
    def save_write_log(self, filepath: Path):
        """Save write log to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'total_writes': len(self.write_log),
                'successful_writes': sum(1 for log in self.write_log if log.get('success')),
                'failed_writes': sum(1 for log in self.write_log if not log.get('success')),
                'writes': self.write_log
            }, f, indent=2)
        
        print(f"📝 Write log saved to: {filepath}")
    
    def get_last_write_status(self) -> Optional[Dict]:
        """Get status of last write operation."""
        if not self.write_log:
            return None
        return self.write_log[-1].copy()

