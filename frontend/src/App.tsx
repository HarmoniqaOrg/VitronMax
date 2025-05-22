import './App.css'; 
import FileUpload from './components/FileUpload';

function App() {
  return (
    <div className="container mx-auto p-4 flex flex-col items-center">
      <header className="mb-8 w-full max-w-4xl">
        <h1 className="text-3xl font-bold text-center">VitronMax Dashboard</h1>
      </header>
      <main className="w-full max-w-md flex justify-center">
        <FileUpload />
      </main>
      <footer className="mt-8 text-center text-sm text-gray-500">
        <p>&copy; {new Date().getFullYear()} VitronMax. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
