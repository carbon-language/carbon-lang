import React from 'react';
import Layout from '@theme/Layout';
import Readme from '../../README.md'
import styles from './index.module.css';

export default function Home() {
  return (
    <Layout>
      <main className={styles.main}>
        <Readme/>
      </main>
    </Layout>
  );
}
